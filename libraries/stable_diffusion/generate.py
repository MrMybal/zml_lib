"""Generate an image with the local ZML Stable Diffusion 1.5 FP32 pipeline."""

import argparse
from contextlib import redirect_stdout
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from prepare import manifest, probe_path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build import scratch_directory
from probe_placement import PLATFORMS, verify_placement


def run_probe(command, backend, count, device, work):
    env = os.environ.copy()
    env.update({key: str(work) for key in ("TMP", "TEMP", "TMPDIR")})
    env["CUDA_CACHE_DISABLE"] = "1"
    result = subprocess.run(command, env=env, capture_output=True, text=True, errors="replace")
    print(result.stderr, end="", file=sys.stderr)
    result.check_returncode()
    return verify_placement(result.stderr, backend, count, device)


def read_tensor(path, shape):
    value = np.fromfile(path, dtype="<f4").reshape(shape)
    if not np.isfinite(value).all():
        raise ValueError(f"Nonfinite output: {path.name}")
    return value


def token_ids(tokenizer, text):
    return tokenizer(text, padding="max_length", max_length=77, truncation=True)["input_ids"]


def encode(args, work, ids, name):
    clip = probe_path(args.bin_dir, "clip", "clip_text_probe")
    destination = work / f"{name}.f32"
    run_probe([clip, args.backend, args.weights / "clip.safetensors", destination, *map(str, ids)],
              args.backend, 1, args.expected_device, work)
    return read_tensor(destination, (77, 768))


def predict(args, work, initial, positive, *, negative=None, timestep=None, image_condition=(), source_latent=None, start=0):
    probe = probe_path(args.bin_dir, "stable_diffusion", "sd15_probe")
    height, width = initial.shape[-2:]
    mode = ("inpaint_" if image_condition else "") + ("sample" if timestep is None else "unet")
    source, target = work / f"{mode}-in.f32", work / f"{mode}-out.f32"
    inputs = [initial, *image_condition]
    if source_latent is not None:
        inputs.append(source_latent)
    inputs.append(positive)
    if negative is not None:
        inputs.append(negative)
    with source.open("wb") as stream:
        for array in inputs:
            np.asarray(array, dtype="<f4").tofile(stream)
    command = [probe, mode, args.backend,
               args.weights / "unet.safetensors", str(height), str(width)]
    if timestep is None:
        command.extend((str(args.steps), str(args.guidance)))
        if image_condition:
            command.append(str(start))
    else:
        command.append(str(timestep))
    count = (args.steps - start) * (1 + int(negative is not None)) if timestep is None else 1
    run_probe([*command, source, target], args.backend, count, args.expected_device, work)
    return read_tensor(target, initial.shape)


def decode(args, work, latent, name="vae"):
    probe = probe_path(args.bin_dir, "stable_diffusion", "sd15_probe")
    height, width = latent.shape[-2:]
    source, target = work / f"{name}-in.f32", work / f"{name}-out.f32"
    np.asarray(latent, dtype="<f4").tofile(source)
    placement = run_probe([probe, "vae", args.backend, args.weights / "vae.safetensors", str(height),
                           str(width), source, target], args.backend, 1, args.expected_device, work)
    return read_tensor(target, (3, height * 8, width * 8)), placement


def initial_noise(seed, height, width):
    return np.random.default_rng(seed).standard_normal((4, height // 8, width // 8), dtype=np.float32)


def image_pixels(rgb):
    return np.rint(np.clip(rgb.transpose(1, 2, 0) / 2 + 0.5, 0, 1) * 255).astype(np.uint8)


def add_options(parser, *, weights_required=True, size_options=True):
    parser.add_argument("--weights", required=weights_required, type=Path)
    parser.add_argument("--bin-dir", required=True, type=Path)
    parser.add_argument("--clip-tokenizer", required=True, type=Path)
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file", type=Path)
    negative = parser.add_mutually_exclusive_group()
    negative.add_argument("--negative-prompt", default="")
    negative.add_argument("--negative-prompt-file", type=Path)
    if size_options:
        parser.add_argument("--width", type=int, default=512)
        parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=PLATFORMS, default="cuda")
    parser.add_argument("--expected-device")
    parser.add_argument("--workspace", type=Path)


def check_options(args, parser, profile="sd15"):
    if args.prompt_file:
        args.prompt = args.prompt_file.read_text(encoding="utf-8")
    if args.negative_prompt_file:
        args.negative_prompt = args.negative_prompt_file.read_text(encoding="utf-8")
    if not args.prompt.strip():
        parser.error("Prompt must contain text.")
    if any(size < 64 or size > 512 or size % 64 for size in (args.width, args.height)):
        parser.error("Width and height must be multiples of 64 between 64 and 512.")
    if not 1 <= args.steps <= 100 or not math.isfinite(args.guidance) or not 1 <= args.guidance <= 20:
        parser.error("Use 1–100 steps and CFG between 1 and 20.")
    if not 0 <= args.seed <= 2147483647:
        parser.error("Seed must be between 0 and 2147483647.")
    for name in ("weights", "bin_dir", "clip_tokenizer"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    if json.loads((args.weights / "sd15.json").read_text(encoding="utf-8")) != manifest(profile):
        parser.error(f"Expected a prepared {profile} FP32 bundle.")


def encode_prompts(args, work):
    from transformers import CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(args.clip_tokenizer, local_files_only=True)
    positive = encode(args, work, token_ids(tokenizer, args.prompt), "positive")
    negative = None
    if args.guidance > 1:
        negative = encode(args, work, token_ids(tokenizer, args.negative_prompt), "negative")
    return positive, negative


def generate(args, work):
    positive, negative = encode_prompts(args, work)
    latent = predict(args, work, initial_noise(args.seed, args.height, args.width), positive, negative=negative)
    rgb, placement = decode(args, work, latent)
    write_png(args.output, image_pixels(rgb))
    return {"file": str(args.output), "width": args.width, "height": args.height, "steps": args.steps,
            "seed": args.seed, "guidance": args.guidance, "scheduler": "ddim", "placement": placement}


def write_png(output, pixels):
    # Encode before creating the exclusive output file.
    from io import BytesIO

    encoded = BytesIO()
    Image.fromarray(pixels).save(encoded, format="PNG")
    with output.open("xb") as stream:
        stream.write(encoded.getbuffer())


def run(args, parser, operation):
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error("Output already exists.")
    with redirect_stdout(sys.stderr):
        if args.workspace:
            with TemporaryDirectory(prefix="sd15-", dir=args.workspace.resolve(strict=True)) as work:
                result = operation(args, Path(work))
        else:
            with scratch_directory() as work:
                result = operation(args, work)
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_options(parser)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check_options(args, parser)
    run(args, parser, generate)


if __name__ == "__main__":
    main()
