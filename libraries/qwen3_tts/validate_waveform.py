"""Compare the shared 12 Hz ZML waveform decoder with Qwen's PyTorch decoder."""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
from safetensors.torch import load_file
import torch

from generate import SAMPLES_PER_FRAME, WAVEFORM_CHUNK, decode_waveform, runner
from prepare import prepare_waveform
from libraries.flux1.prepare import schema_for


def reference(model, device):
    from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2DecoderConfig
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Decoder

    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    decoder = Qwen3TTSTokenizerV2DecoderConfig(**config["decoder_config"])
    decoder._attn_implementation = "eager"
    decoder.use_cache = False
    graph = Qwen3TTSTokenizerV2Decoder(decoder)
    graph.load_state_dict({key.removeprefix("decoder."): tensor
                           for key, tensor in load_file(model / "model.safetensors").items()
                           if key.startswith("decoder.")}, strict=True)
    return graph.eval().to(device=device, dtype=torch.float32)


def fixture(path):
    if path is not None:
        values = np.load(path, allow_pickle=False)
        if (values.ndim != 2 or values.shape[1] != 16 or not 1 <= len(values) <= 1024
                or values.dtype.kind not in "iu" or values.min() < 0 or values.max() >= 2048):
            raise ValueError("Expected one to 1024 rows of 16 integer codes between 0 and 2047")
        return ((path.name, values.astype(np.int64, copy=False)),)

    rng = np.random.default_rng(712)
    # Cover single-frame overlap, the local attention boundary and the chunk splice.
    return tuple((str(length), rng.integers(0, 2048, size=(length, 16), dtype=np.int64))
                 for length in (1, 73, 301))


def compare(expected, actual, label):
    if expected.ndim != 1 or actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError(f"{label}: invalid waveform shape or values")
    if not np.isfinite(expected).all() or np.sqrt(np.mean(expected.astype(np.float64) ** 2)) < 1e-6:
        raise ValueError(f"{label}: invalid reference waveform")
    difference = actual.astype(np.float64) - expected
    peak = float(np.abs(difference).max())
    rmse = float(np.sqrt(np.mean(difference ** 2)))
    if rmse > 0.001:
        raise ValueError(f"{label}: RMSE {rmse} exceeds 0.001 (peak {peak})")
    return {"rmse": rmse, "max_abs": peak}


def validate(args, work):
    if args.waveform_weights is None:
        args.waveform_weights = work / "waveform.safetensors"
        schema = schema_for(args.probe, "waveform")
        prepare_waveform(args.model, schema, args.waveform_weights)
    if args.reference_device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    decoder = reference(args.model, args.reference_device)
    cases = []
    for label, codes in fixture(args.codes):
        inputs = torch.from_numpy(codes.T.copy()).unsqueeze(0).to(args.reference_device)
        with torch.inference_mode():
            expected = decoder.chunked_decode(inputs).flatten().cpu().numpy()
        samples = len(codes) * SAMPLES_PER_FRAME
        if expected.shape != (samples,):
            raise ValueError(f"{label}: expected exactly {samples} samples, got {expected.shape}")
        if len(codes) <= WAVEFORM_CHUNK:
            with runner(args, work, "waveform", len(codes), (samples,)) as infer:
                actual = infer(codes)
                repeated = infer(codes)
            if not np.array_equal(actual, repeated):
                raise ValueError(f"{label}: repeated native inference changed the waveform")
            repeats = 2
        else:
            actual = decode_waveform(args, work, codes)
            repeats = 1
        cases.append({"case": label, "frames": len(codes), "samples": samples, "repeats": repeats,
                      **compare(expected, actual, label)})
    return {"backend": args.backend, "device": args.expected_device, "pjrt": args.expected_pjrt,
            "reference_device": args.reference_device, "torch_version": torch.__version__, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "probe", "scratch"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--waveform-weights", type=Path, help="reuse an already prepared waveform checkpoint")
    parser.add_argument("--codes", type=Path, help="use codes saved by generate.py; otherwise test fixed seeded fixtures")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--expected-device", required=True)
    parser.add_argument("--expected-pjrt")
    parser.add_argument("--reference-device", default="cpu")
    args = parser.parse_args()
    args.scratch = args.scratch.resolve(strict=True)
    if args.scratch.drive.upper() == "C:":
        parser.error("--scratch must be outside C:")
    with redirect_stdout(sys.stderr), TemporaryDirectory(prefix="qwen12-waveform-validation-", dir=args.scratch) as folder:
        result = validate(args, Path(folder))
    if args.output:
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2)
            output.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
