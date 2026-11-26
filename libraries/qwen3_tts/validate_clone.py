"""Validate Qwen3-TTS Base voice cloning in ZML against official Qwen PyTorch."""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from safetensors import safe_open
import torch
from transformers import AutoTokenizer

import generate
from prepare import PROFILES, verify_config, verify_speaker_config
from speaker import FFT, HOP, MELS, RATE, log_mels, read_reference
from validate_waveform import compare, reference


def reference_speaker(path, device):
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSSpeakerEncoder

    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    model = Qwen3TTSSpeakerEncoder(Qwen3TTSSpeakerEncoderConfig(**config["speaker_encoder_config"]))
    with safe_open(path / "model.safetensors", framework="pt", device="cpu") as checkpoint:
        model.load_state_dict({name: checkpoint.get_tensor(f"speaker_encoder.{name}") for name in model.state_dict()})
    return model.eval().to(device=device, dtype=torch.float32)


def error(expected, actual, label, threshold):
    if expected.shape != actual.shape or not expected.size or not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError(f"{label}: invalid or mismatched output shape")
    difference = actual.astype(np.float64) - expected
    rmse = float(np.sqrt(np.mean(difference ** 2)))
    if rmse > threshold:
        raise ValueError(f"{label}: RMSE {rmse} exceeds {threshold}")
    return {"rmse": rmse, "max_abs": float(np.abs(difference).max())}


def cosine(left, right):
    left = left.astype(np.float64).ravel()
    right = right.astype(np.float64).ravel()
    norms = np.linalg.norm(left) * np.linalg.norm(right)
    if norms <= 0:
        raise ValueError("The speaker embedding has zero norm")
    return float(np.clip(np.dot(left, right) / norms, -1, 1))


def check_speaker(args, work, path, model):
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

    samples = read_reference(path)
    mels = log_mels(samples)
    with torch.inference_mode():
        official_mels = mel_spectrogram(
            torch.from_numpy(samples)[None], n_fft=FFT, num_mels=MELS, sampling_rate=RATE,
            hop_size=HOP, win_size=FFT, fmin=0, fmax=RATE // 2,
        ).transpose(1, 2).contiguous()
        expected = model(official_mels.to(args.reference_device)).cpu().numpy()
    frontend = error(official_mels[0].numpy(), mels, "Log-mels", 0.0001)
    width = PROFILES[args.profile][1]
    with generate.runner(args, work, "speaker", len(mels), (1, width)) as encode:
        actual = encode(mels)
        repeated = encode(mels)
    if not np.array_equal(actual, repeated):
        raise ValueError("Repeated speaker inference changed the embedding")
    embedding = error(expected, actual, "Speaker embedding", 0.001)
    embedding["cosine"] = cosine(expected, actual)
    return expected[0], actual[0], {
        "audio": path.name, "frames": len(mels), "dimensions": width, "log_mels": frontend,
        "embedding": embedding, "repeats": 2,
    }


def reference_audio_encoder(path, device):
    from transformers import MimiConfig
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Encoder

    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    voice = MimiConfig(**config["encoder_config"])
    voice._attn_implementation = "eager"
    model = Qwen3TTSTokenizerV2Encoder(voice)
    with safe_open(path / "model.safetensors", framework="pt", device="cpu") as checkpoint:
        model.load_state_dict({name: checkpoint.get_tensor(f"encoder.{name}") for name in model.state_dict()})
    return model.eval().to(device=device, dtype=torch.float32)


def check_audio_encoding(args, work, model, samples):
    with torch.inference_mode():
        expected = model.encode(
            torch.from_numpy(samples)[None, None].to(args.reference_device), num_quantizers=16, return_dict=True,
        ).audio_codes[0].T.contiguous().cpu().numpy()
    actual = generate.encode_reference(args, work, samples)
    if not np.array_equal(expected, actual):
        if expected.shape != actual.shape:
            raise ValueError(f"Audio encoder shape mismatch: {actual.shape}, expected {expected.shape}")
        first = tuple(np.argwhere(expected != actual)[0])
        raise ValueError(f"Different reference code at frame {first[0]}, group {first[1]}: "
                         f"{actual[first]}, expected {expected[first]}")
    return expected, actual, {"samples": len(samples), "frames": len(actual), "groups": 16, "codes_equal": True}


def reference_codes(args, xvector, audio_codes=None):
    from accelerate import init_empty_weights
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration

    config = Qwen3TTSConfig.from_pretrained(args.model, local_files_only=True)
    config._attn_implementation = "eager"
    with init_empty_weights():
        model = Qwen3TTSForConditionalGeneration(config)
    weights = {}
    with safe_open(args.model / "model.safetensors", framework="pt", device="cpu") as checkpoint:
        for key in model.state_dict():
            tensor = checkpoint.get_tensor(key)
            if key.startswith("talker."):
                # Match the storage conversion applied by the ZML launcher.
                tensor = tensor.to(torch.float16)
            weights[key] = tensor.to(device=args.reference_device, dtype=torch.float32)
    model.load_state_dict(weights, strict=True, assign=True)
    model.eval().to(args.reference_device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    message = f"<|im_start|>assistant\n{args.text}<|im_end|>\n<|im_start|>assistant\n"
    ids = tokenizer(message, add_special_tokens=False, return_tensors="pt").input_ids.to(args.reference_device)
    icl = audio_codes is not None
    ref_ids = None
    if icl:
        message = f"<|im_start|>assistant\n{args.reference_text}<|im_end|>\n"
        ref_ids = [tokenizer(message, add_special_tokens=False, return_tensors="pt").input_ids.to(args.reference_device)]
    prompt = {
        "ref_code": [torch.from_numpy(audio_codes) if icl else None], "ref_spk_embedding": [torch.from_numpy(xvector)],
        "x_vector_only_mode": [not icl], "icl_mode": [icl],
    }
    random = np.random.default_rng(args.seed)

    def multinomial(probabilities, num_samples, **_):
        if probabilities.shape[0] != 1 or num_samples != 1:
            raise ValueError("The paired reference expects a single draw")
        index = generate.draw_code(probabilities[0].cpu().numpy(), random)
        return torch.tensor([[index]], dtype=torch.long, device=probabilities.device)

    with patch.object(torch, "multinomial", side_effect=multinomial), \
            patch.object(model.talker, "generate", wraps=model.talker.generate) as talker:
        frames, _ = model.generate(
            input_ids=[ids], languages=[args.language], voice_clone_prompt=prompt, ref_ids=ref_ids, non_streaming_mode=True,
            do_sample=True, top_k=50, top_p=1.0, temperature=0.9,
            subtalker_dosample=True, subtalker_top_k=50, subtalker_top_p=1.0, subtalker_temperature=0.9,
            repetition_penalty=1.05, max_new_tokens=args.max_frames + 1,
        )
        prefix = talker.call_args.kwargs["inputs_embeds"][0].detach().cpu().numpy()
    return frames[0].cpu().numpy(), prefix


def validate(args, work):
    if args.reference_device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model = reference_speaker(args.model, args.reference_device)
    expected, actual, primary = check_speaker(args, work, args.reference_audio, model)
    report = {"speaker": primary}
    if args.alternate_audio:
        alt_expected, alt_actual, alternate = check_speaker(args, work, args.alternate_audio, model)
        alternate["cosine_to_primary"] = cosine(actual, alt_actual)
        alternate["reference_cosine_to_primary"] = cosine(expected, alt_expected)
        if np.array_equal(actual, alt_actual):
            raise ValueError("Changing the reference audio did not change the speaker embedding")
        report["alternate"] = alternate
    del model
    if args.reference_device.startswith("cuda"):
        torch.cuda.empty_cache()

    reference_ref, zml_ref = None, None
    if args.reference_text:
        encoder = reference_audio_encoder(args.codec, args.reference_device)
        reference_ref, zml_ref, report["audio_encoder"] = check_audio_encoding(
            args, work, encoder, read_reference(args.reference_audio),
        )
        if args.alternate_audio:
            # Exercise the non-aligned tail padding with a second recording.
            samples = read_reference(args.alternate_audio)
            if len(samples) % generate.SAMPLES_PER_FRAME == 0:
                samples = samples[:-1]
            _, _, report["audio_encoder_alternate"] = check_audio_encoding(args, work, encoder, samples)
        del encoder
        if args.reference_device.startswith("cuda"):
            torch.cuda.empty_cache()

    prefixes = []
    build_prompt = generate.prompt

    def capture(*items):
        prefix, pad = build_prompt(*items)
        prefixes.append(prefix)
        return prefix, pad

    with patch.object(generate, "prompt", side_effect=capture):
        frames, complete = generate.codes(args, work, zml_ref)
    if not complete or not len(frames):
        raise ValueError("The ZML clone did not finish within the frame budget")
    audio, _ = generate.clone_waveform(args, work, frames, zml_ref)
    reference_frames, prefix = reference_codes(args, expected, reference_ref)
    report["prompt"] = error(prefix, prefixes[0], "Talker prompt", 0.001)
    if not np.array_equal(reference_frames, frames):
        common = min(len(reference_frames), len(frames))
        difference = np.argwhere(reference_frames[:common] != frames[:common])
        first = difference[0].tolist() if len(difference) else None
        raise ValueError(f"Different sampled codes: ZML {len(frames)} frames, PyTorch {len(reference_frames)}; "
                         f"first different frame and group: {first}")
    if args.reference_device.startswith("cuda"):
        torch.cuda.empty_cache()
    decoder = reference(args.codec, args.reference_device)
    decode_codes = reference_frames if reference_ref is None else np.concatenate((reference_ref, reference_frames))
    with torch.inference_mode():
        inputs = torch.from_numpy(decode_codes.T.copy()).unsqueeze(0).to(args.reference_device)
        expected_audio = decoder.chunked_decode(inputs).flatten().cpu().numpy()
    if reference_ref is not None:
        expected_audio = expected_audio[len(reference_ref) * generate.SAMPLES_PER_FRAME:]
    report.update(profile=args.profile, text=args.text, language=args.language, frames=len(frames),
                  seed=args.seed, voice_mode="icl" if zml_ref is not None else "x_vector_only",
                  sampled_codes_equal=True, sample_rate=RATE, samples=len(audio), eos=complete,
                  backend=args.backend, device=args.expected_device, pjrt=args.expected_pjrt,
                  reference_device=args.reference_device, torch_version=torch.__version__,
                  waveform=compare(expected_audio, audio, "Clone waveform"))
    if args.wav:
        generate.save_wave(args.wav, audio, RATE)
    if args.reference_wav:
        generate.save_wave(args.reference_wav, expected_audio, RATE)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("probe", "model", "prepared", "waveform-weights", "codec", "reference-audio", "scratch"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("alternate-audio", "audio-encoder-weights", "wav", "reference-wav", "output"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--profile", required=True, choices=PROFILES)
    parser.add_argument("--text", required=True)
    parser.add_argument("--reference-text", default="")
    parser.add_argument("--language", default="auto", type=str.lower)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max-frames", default=64, type=int)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--expected-device", required=True)
    parser.add_argument("--expected-pjrt")
    parser.add_argument("--reference-device", default="cpu")
    args = parser.parse_args()
    args.reference_text = args.reference_text.strip()
    if not args.text.strip() or any(mark in args.text + args.reference_text for mark in ("<|im_start|>", "<|im_end|>")):
        parser.error("Enter text without reserved chat markers")
    if not 1 <= args.max_frames <= 1024:
        parser.error("--max-frames must be between 1 and 1024")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if bool(args.reference_text) != bool(args.audio_encoder_weights):
        parser.error("ICL requires both --reference-text and --audio-encoder-weights")
    args.scratch = args.scratch.resolve(strict=True)
    if args.scratch.drive.upper() == "C:":
        parser.error("--scratch must be outside C:")
    args.speaker = args.instruction = ""
    verify_config(args.model, args.profile)
    verify_speaker_config(args.model, args.profile)
    with redirect_stdout(sys.stderr), TemporaryDirectory(prefix="qwen-base-validation-", dir=args.scratch) as folder:
        report = validate(args, Path(folder))
    if args.output:
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(report, output, indent=2, ensure_ascii=False)
            output.write("\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
