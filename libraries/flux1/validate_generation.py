"""Validate CUDA FLUX.1 Schnell generation and sample device-wide memory use."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build import scratch_directory
from probe_placement import verify_placement


HERE = Path(__file__).resolve().parent


def repack_scaled(checkpoint, bin_dir, output):
    """Create a lossless ComfyUI-style scaled test file from the pinned ordinary FP8 Schnell."""
    import torch
    from safetensors import safe_open

    from prepare import denoiser_prefix, probe_path, schema_for, write_tensors

    torch.set_num_threads(4)
    schema = schema_for(probe_path(bin_dir, "flux1", "flux1_probe"), "schnell", "e4m3")
    with safe_open(checkpoint, framework="pt", device="cpu") as source:
        prefix = denoiser_prefix(source)
        fields = {"scaled_fp8": ((2,), "f32")}
        for name, shape in schema.items():
            matrix = name.endswith(".weight") and len(shape) == 2
            fields[name] = (shape, "e4m3" if matrix else "f32")
            if matrix:
                fields[name.removesuffix(".weight") + ".scale_weight"] = ((), "f32")

        non_unit = 0

        def values():
            nonlocal non_unit
            yield torch.zeros(2, dtype=torch.float32)
            for index, name in enumerate(schema):
                original = source.get_tensor(prefix + name)
                if original.dtype != torch.float8_e4m3fn:
                    raise ValueError(f"The lossless fixture expects the ordinary E4M3FN checkpoint: {name}")
                value = original.float()
                if fields[name][1] == "f32":
                    yield value
                    continue
                expansion = 2 ** (1 + index % 3)
                high = value.abs().max().item()
                while high * expansion > 448 and expansion > 1:
                    expansion //= 2
                encoded = (value * expansion).to(torch.float8_e4m3fn)
                if not torch.equal(encoded.float() / expansion, value):
                    raise ValueError(f"Lossless FP8 repacking failed for {name}")
                non_unit += int(expansion != 1)
                yield encoded.view(torch.uint8)
                yield torch.tensor(1 / expansion, dtype=torch.float32)

        write_tensors(output, "validation_e4m3_scaled", fields, values())
        return non_unit


def image_size(value):
    try:
        width, height = map(int, value.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use WIDTHxHEIGHT, for example 512x512.") from exc
    if any(size < 16 or size > 1024 or size % 16 for size in (width, height)):
        raise argparse.ArgumentTypeError("Dimensions must be multiples of 16 between 16 and 1024.")
    return width, height


def local_environment(work):
    env = os.environ.copy()
    env.update({name: str(work) for name in ("TEMP", "TMP", "TMPDIR", "HF_HOME")})
    env.update(CUDA_CACHE_DISABLE="1", PYTHONDONTWRITEBYTECODE="1",
               TORCHINDUCTOR_CACHE_DIR=str(work / "inductor"))
    return env


def measure(command, env, work, gpu):
    smi = ["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    initial = int(subprocess.check_output(smi, env=env, text=True).strip())
    path = work / "gpu-memory.txt"
    with path.open("w", encoding="ascii") as stream:
        sampler = subprocess.Popen([*smi, "--loop-ms=200"], env=env, stdout=stream)
        start = monotonic()
        try:
            result = subprocess.run(command, env=env, capture_output=True, encoding="utf-8", errors="replace")
        finally:
            seconds = round(monotonic() - start, 2)
            sampler.terminate()
            sampler.wait()
    samples = [int(line) for line in path.read_text(encoding="ascii").splitlines() if line.strip().isdigit()]
    if not samples:
        raise RuntimeError("nvidia-smi did not record any GPU memory samples.")
    peak = max(initial, max(samples))
    return result, {"seconds": seconds, "initial_gpu_used_mib": initial, "peak_gpu_used_mib": peak,
                    "peak_gpu_delta_mib": peak - initial, "gpu_samples": len(samples)}


def validate(args, weights, size, work):
    width, height = size
    output = work / "flux.png"
    command = [sys.executable, "-B", HERE / "generate.py", "--backend", "cuda", "--bin-dir", args.bin_dir,
               "--weights", weights, "--clip-tokenizer", args.clip_tokenizer, "--t5-tokenizer", args.t5_tokenizer,
               "--prompt", args.prompt, "--output", output, "--decoder", args.decoder,
               "--expected-device", args.expected_device, "--seed", str(args.seed), "--steps", str(args.steps),
               "--text-tokens", str(args.text_tokens), "--width", str(width), "--height", str(height),
               "--workspace", work]
    result, metrics = measure(command, local_environment(work), work, args.gpu)
    report = {"width": width, "height": height, "seed": args.seed, "steps": args.steps, "decoder": args.decoder,
              "text_tokens": args.text_tokens, **metrics}
    if result.returncode:
        print(result.stderr[-8000:], file=sys.stderr)
        return {**report, "status": "failed", "exit_code": result.returncode}
    image = json.loads(result.stdout)
    placement = verify_placement(result.stderr, "cuda", args.steps + 4, args.expected_device)
    if image["placement"] != placement or Path(image["file"]) != output:
        raise RuntimeError("The reported placement or output path does not match this test.")
    if any(image[key] != report[key] for key in ("width", "height", "seed", "steps", "decoder")):
        raise RuntimeError("The reported generation settings do not match this test.")
    with Image.open(output) as png:
        if (png.format, png.mode, png.size) != ("PNG", "RGB", size):
            raise ValueError("Unexpected FLUX.1 image format, mode or dimensions.")
        png.verify()
    with Image.open(output) as png:
        if all(low == high for low, high in png.getextrema()):
            raise ValueError("FLUX.1 produced a constant image.")
    with output.open("rb") as stream:
        sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    return {**report, "status": "passed", "placement": placement, "storage": image["storage"], "png_sha256": sha256}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--weights", type=Path, help="Existing prepared Schnell bundle.")
    source.add_argument("--checkpoint", type=Path, help="Local combined ComfyUI checkpoint to prepare temporarily.")
    extra = parser.add_mutually_exclusive_group()
    extra.add_argument("--denoiser", type=Path, help="Separate Schnell denoiser to use with --checkpoint.")
    extra.add_argument("--repack-scaled", action="store_true", help="Losslessly repack the ordinary Schnell FP8 checkpoint as a scaled fixture.")
    parser.add_argument("--taef1", type=Path, help="Local preview decoder checkpoint for bundle preparation.")
    for name in ("bin-dir", "clip-tokenizer", "t5-tokenizer"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-device", required=True, help="Substring of the PJRT CUDA device name.")
    parser.add_argument("--gpu", default="0", help="nvidia-smi index or UUID of the CUDA device to measure.")
    parser.add_argument("--size", type=image_size, action="append", help="Repeat to test multiple image sizes.")
    parser.add_argument("--prompt", default="A red lighthouse by the sea at dawn.")
    parser.add_argument("--decoder", choices=("vae", "taef1"), default="vae")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--text-tokens", type=int, default=256)
    args = parser.parse_args()
    if not 1 <= args.steps <= 50 or not 2 <= args.text_tokens <= 512 or args.seed < 0:
        parser.error("Use 1 to 50 steps, 2 to 512 text tokens and a nonnegative seed.")
    if args.checkpoint is not None and args.decoder == "taef1" and args.taef1 is None:
        parser.error("TAEF1 validation with --checkpoint also requires --taef1.")
    if (args.denoiser or args.repack_scaled) and args.checkpoint is None:
        parser.error("--denoiser and --repack-scaled require --checkpoint.")

    sizes = args.size or [(512, 512), (1024, 1024)]
    results = []
    with scratch_directory() as work:
        weights = args.weights
        if args.checkpoint is not None:
            weights = work / "bundle"
            denoiser = args.denoiser
            if args.repack_scaled:
                denoiser = work / "comfy-scaled-fixture.safetensors"
                with scratch_directory() as conversion:
                    keys = ("TMP", "TEMP", "TMPDIR", "TORCHINDUCTOR_CACHE_DIR")
                    original = {key: os.environ.get(key) for key in keys}
                    try:
                        os.environ.update({key: str(conversion) for key in keys})
                        count = repack_scaled(args.checkpoint, args.bin_dir, denoiser)
                    finally:
                        for key, value in original.items():
                            if value is None:
                                os.environ.pop(key, None)
                            else:
                                os.environ[key] = value
                if count == 0:
                    raise ValueError("The scaled fixture has no non-unit scales.")
                print(f"Repacked {count} weights with non-unit scales", file=sys.stderr)
            command = [sys.executable, "-B", HERE / "prepare.py", "--source", args.checkpoint,
                       "--bin-dir", args.bin_dir, "--output", weights]
            if denoiser is not None:
                command.extend(["--denoiser", denoiser])
            if args.taef1 is not None:
                command.extend(["--taef1", args.taef1])
            subprocess.run(command, env=local_environment(work), check=True)
        for size in sizes:
            with scratch_directory() as case:
                report = validate(args, weights, size, case)
                print(json.dumps(report, allow_nan=False), flush=True)
                results.append(report["status"] == "passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
