"""Edit one image with the local Qwen Image Edit 2511 ZML pipeline."""

import argparse
from contextlib import contextmanager, redirect_stdout
import json
import math
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageOps

from pipeline import edit_prompt, encode_conditioning, run_denoiser, vae_run
from prepare import prepare, schema_for
from scripts.build import scratch_directory


MODEL_FILES = {
    "denoiser": "diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors",
    "vae": "vae/qwen_image_vae.safetensors",
    "encoder": "text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
}


def schedule(steps, shift):
    # The simple scheduler selects descending indices from 1,000 shifted flow times.
    ticks = 1000 - np.arange(steps) * 1000 // steps
    times = ticks.astype(np.float32) / np.float32(1000)
    times = shift * times / (1 + (shift - 1) * times)
    times = np.concatenate((times, np.zeros(1, dtype=np.float32)))
    if not np.isfinite(times).all() or times[0] > 1 or not np.all(np.diff(times) < 0):
        raise ValueError("Choose a shift that gives strictly decreasing FP32 timesteps")
    return times


@contextmanager
def work_directory(parent):
    if parent is None:
        with scratch_directory() as work:
            yield work
    else:
        parent = parent.resolve(strict=True)
        if parent.drive.upper() == "C:":
            raise ValueError("Choose a temporary workspace outside C:")
        with TemporaryDirectory(prefix="qwen-edit-", dir=parent) as work:
            yield Path(work)


def generate(args, work):
    with Image.open(args.image) as source:
        image = ImageOps.exif_transpose(source).convert("RGB").resize(
            (args.width, args.height), Image.Resampling.LANCZOS,
        )
    prompt = edit_prompt(args.prompt)
    print("Encoding the reference image and edit prompt", file=sys.stderr, flush=True)
    text = encode_conditioning(args, work, image, (args.vision_size, args.vision_size), prompt)
    encode_path = work / "vae-encode.safetensors"
    decode_path = work / "vae-decode.safetensors"
    for mode, path in (("vae_encode", encode_path), ("vae_decode", decode_path)):
        prepare(args.vae, path, schema_for(args.probe, mode), mode)
    rgb = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / np.float32(127.5) - np.float32(1)
    h, w = args.height // 8, args.width // 8
    reference, _ = vae_run(args, work, "vae_encode", encode_path, rgb, h, w, "reference")
    noise = np.random.default_rng(args.seed).standard_normal(reference.shape, dtype=np.float32)
    mask = np.ones(len(text), dtype=np.int64)

    profile = "edit_2511"
    weights = work / "denoiser.safetensors"
    print("Preparing the FP8 denoiser checkpoint", file=sys.stderr, flush=True)
    prepare(args.denoiser, weights, schema_for(args.probe, profile), profile)
    print(f"Running {args.steps} Euler steps", file=sys.stderr, flush=True)
    latent, denoiser_placement = run_denoiser(
        args, work, profile, weights, noise, [reference], text, mask, args.times, "edit",
    )
    rgb, placement = vae_run(args, work, "vae_decode", decode_path, latent, h, w, "result")
    pixels = np.rint(np.clip(rgb.transpose(1, 2, 0) * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8)
    with args.output.open("xb") as output:
        try:
            Image.fromarray(pixels).save(output, format="PNG")
        except BaseException:
            output.close()
            args.output.unlink()
            raise
    return {
        "file": str(args.output), "profile": profile, "width": args.width, "height": args.height,
        "seed": args.seed, "steps": args.steps, "scheduler": "simple-euler", "shift": args.shift,
        "placement": placement, "denoiser_placement": denoiser_placement,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file", type=Path)
    for name in ("probe", "encoder-probe", "model-root", "tokenizer", "image", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--backend", choices=("cuda",), default="cuda")
    parser.add_argument("--expected-device")
    parser.add_argument("--expected-pjrt")
    parser.add_argument("--cuda-low-memory", action="store_true")
    parser.add_argument("--cuda-math-library", type=Path)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--vision-size", type=int, default=224)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--shift", type=float, default=1.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if any(size < 16 or size > 1024 or size % 16 for size in (args.width, args.height)):
        parser.error("Width and height must be multiples of 16 in 16..1024")
    if args.vision_size <= 0 or args.vision_size % 28 or (args.vision_size // 14) ** 2 > 2048:
        parser.error("Vision size must be a multiple of 28 within the 2,048-patch limit")
    if not 1 <= args.steps <= 60 or args.seed < 0 or not math.isfinite(args.shift) or args.shift <= 0:
        parser.error("Use 1 to 60 steps, a nonnegative seed and a finite positive shift")
    if args.cuda_low_memory and sys.platform != "linux":
        parser.error("CUDA low-memory execution is available on Linux")
    if args.cuda_math_library and not args.cuda_low_memory:
        parser.error("The CUDA math library requires --cuda-low-memory")
    if args.cuda_math_library:
        os.environ["ZML_QWEN_CUDA_MATH"] = str(args.cuda_math_library.resolve(strict=True))
    if 2 * (args.width // 16) * (args.height // 16) > 4096 and (
        not args.cuda_low_memory or not os.getenv("ZML_QWEN_CUDA_MATH")
    ):
        parser.error("More than 4,096 packed image tokens require the Linux CUDA reference-math mode")
    try:
        args.times = schedule(args.steps, args.shift)
    except ValueError as error:
        parser.error(str(error))
    if args.prompt_file:
        args.prompt = args.prompt_file.read_text(encoding="utf-8-sig")
    if not args.prompt.strip() or any(token in args.prompt for token in (
        "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|image_pad|>", "<|vision_end|>",
    )):
        parser.error("Enter an edit prompt without reserved Qwen chat tokens")
    if (args.output.suffix.lower() != ".png" or args.output.exists() or args.output.is_symlink()
            or not args.output.parent.is_dir()):
        parser.error("Choose a new PNG path in an existing directory")
    args.output = args.output.resolve()
    for key in ("probe", "encoder_probe", "tokenizer", "image"):
        setattr(args, key, getattr(args, key).resolve(strict=True))
    for key, relative in MODEL_FILES.items():
        setattr(args, key, (args.model_root / relative).resolve(strict=True))
    with redirect_stdout(sys.stderr), work_directory(args.workspace) as work:
        os.environ.update({key: str(work) for key in ("TEMP", "TMP", "TMPDIR", "HF_HOME", "XDG_CACHE_HOME")})
        os.environ.update(CUDA_CACHE_DISABLE="1", HF_HUB_OFFLINE="1")
        if os.environ.get("ZML_QWEN_CUDA_MATH"):
            os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
            os.environ.setdefault("ZML_QWEN_DENOISER_XLA_FLAGS",
                                  "--xla_gpu_enable_triton_gemm=false --xla_gpu_enable_cublaslt=false")
        report = generate(args, work)
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
