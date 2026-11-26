"""Validate the Phonon neural pipeline and timestamps against PyTorch."""

import argparse
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np
from safetensors.torch import load_file
import torch
from transformers import ParakeetFeatureExtractor
from transformers.models.parakeet.configuration_parakeet import ParakeetEncoderConfig
from transformers.models.parakeet.modeling_parakeet import ParakeetEncoder

from transcribe import BLANK, MELS, RATE, format_text, invoke, log_mels, read_audio, transcribe
from validate_tdt import step


def rmse(actual, expected):
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError(f"Invalid arrays for comparison {actual.shape}, {expected.shape}")
    return float(np.sqrt(np.square(actual.astype(np.float64) - expected).mean()))


def reference_encoder(args, features):
    spec = json.loads(args.config.read_text(encoding="utf-8"))["encoder_config"]
    config = ParakeetEncoderConfig.from_dict(spec)
    config._attn_implementation = "eager"
    weights = {}
    for name, value in load_file(args.encoder, device="cpu").items():
        if name.endswith(("pointwise_conv1.weight", "pointwise_conv2.weight")):
            value = value.unsqueeze(-1)
        weights[name.removeprefix("encoder.")] = value
    model = ParakeetEncoder(config).eval()
    missing, unexpected = model.load_state_dict(weights, strict=False, assign=True)
    if unexpected or any(not name.endswith(".num_batches_tracked") for name in missing):
        raise ValueError(f"PyTorch encoder schema mismatch {missing}, {unexpected}")
    del weights
    model.to(args.reference_device)
    features = torch.from_numpy(features).unsqueeze(0).to(args.reference_device)
    return model(features).last_hidden_state[0].cpu().numpy()


def reference_decode(args, features):
    weights = load_file(args.tdt, device=args.reference_device)
    lstm = torch.nn.LSTM(640, 640, num_layers=2, batch_first=True, device=args.reference_device).eval()
    prefix = "decoder.lstm."
    lstm.load_state_dict({name.removeprefix(prefix): value for name, value in weights.items() if name.startswith(prefix)})
    frames = torch.from_numpy(features).to(args.reference_device)
    state = torch.zeros((4, 640), device=args.reference_device)
    events, token, frame = [], BLANK, 0
    while frame < len(frames):
        if len(events) >= len(frames) * 10:
            raise RuntimeError("Reference TDT did not advance")
        scores = step(weights, lstm, token, frames[frame], state)
        current = int(scores[:8193].argmax())
        duration = int(scores[8193:8198].argmax())
        if current == BLANK:
            duration = max(1, duration)
        else:
            state = scores[8198:].reshape(4, 640)
            token = current
        events.append({"token": current, "frame": frame, "duration": duration})
        frame += duration
    return events


def word_errors(actual, expected):
    guess, truth = (re.findall(r"[^\W_]+", value.casefold()) for value in (actual, expected))
    previous = list(range(len(guess) + 1))
    for row, target in enumerate(truth, 1):
        current = [row]
        for col, token in enumerate(guess, 1):
            current.append(min(previous[col] + 1, current[col - 1] + 1, previous[col - 1] + (token != target)))
        previous = current
    return previous[-1], len(truth)


def validate(args, work):
    waveform, seconds = read_audio(args.audio)
    features = log_mels(waveform)
    extract = ParakeetFeatureExtractor(feature_size=MELS)
    batch = extract(waveform, sampling_rate=RATE, return_tensors="pt")
    torch_mels = batch.input_features[0, batch.attention_mask[0].bool()].numpy()
    mel_error = rmse(features, torch_mels)
    print(json.dumps({"stage": "log_mel", "shape": features.shape, "rmse": mel_error}), flush=True)
    if mel_error > 0.0002:
        raise ValueError("Log-mel frontend differs from the reference")
    expected = reference_encoder(args, features)
    events = reference_decode(args, expected)
    ref_text = format_text(events, args.tokenizer, seconds)
    if args.reference_device.startswith("cuda"):
        torch.cuda.empty_cache()
    packed, output = work / "features.f32", work / "encoder.f32"
    features.astype("<f4", copy=False).tofile(packed)
    place = (args.probe, args.backend, args.expected_device, args.expected_pjrt)
    encoder_placement = invoke(*place, "encode", args.encoder, str(len(features)), packed, output)
    native = np.fromfile(output, dtype="<f4").reshape((-1, 1024))
    metric = rmse(native, expected)
    print(json.dumps({"stage": "encoder", "shape": native.shape, "rmse": metric,
                      "max_abs": float(np.abs(native - expected).max()), "placement": encoder_placement}), flush=True)
    if metric > 0.001:
        raise ValueError("Encoder differs from the PyTorch reference")

    report = transcribe(args, work)
    errors = None
    if args.expected_text:
        truth = args.expected_text.read_text(encoding="utf-8").strip()
        edits, words = word_errors(report["text"], truth)
        if words == 0:
            raise ValueError("Expected text is empty")
        errors = edits / words
    print(json.dumps({"stage": "asr", "report": report, "reference_text": ref_text["text"],
                      "reference_tokens": ref_text["tokens"], "word_error_rate": errors}), flush=True)
    native_ids = [token["id"] for token in report["tokens"]]
    expected_ids = [token["id"] for token in ref_text["tokens"]]
    if not native_ids or native_ids != expected_ids or report["text"] != ref_text["text"]:
        raise ValueError("Transcription tokens differ from the reference")
    maximum = max(abs(x[key] - y[key]) for x, y in zip(report["tokens"], ref_text["tokens"], strict=True)
                  for key in ("start", "end"))
    if maximum > 0.081:
        raise ValueError(f"Token timing differs by {maximum:.4f} seconds")
    if not report["words"] or any(not 0 <= word["start"] <= word["end"] <= report["seconds"] for word in report["words"]):
        raise ValueError("Invalid word timestamps")
    print(json.dumps({"stage": "timestamps", "max_seconds": maximum, "tokens": len(native_ids)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("probe", "encoder", "tdt", "audio", "tokenizer", "scratch", "config"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--expected-text", type=Path)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--expected-device", required=True)
    parser.add_argument("--expected-pjrt")
    parser.add_argument("--reference-device", default="cpu")
    args = parser.parse_args()
    args.scratch = args.scratch.resolve(strict=True)
    if args.scratch.drive.upper() == "C:":
        parser.error("--scratch must be outside C:")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with torch.inference_mode(), TemporaryDirectory(prefix="phonon-validation-", dir=args.scratch) as temp:
        validate(args, Path(temp))


if __name__ == "__main__":
    main()
