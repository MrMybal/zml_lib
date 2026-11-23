"""Run local FLUX.1 Schnell text-to-image with the ZML denoiser and decoder."""

import argparse
from contextlib import contextmanager, redirect_stdout
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from lora import prepare_loras
from fp8 import FORMAT as SCALED_FP8
from prepare import probe_path, schema_for

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
    array = np.fromfile(path, dtype="<f4").reshape(shape)
    if not np.isfinite(array).all():
        raise ValueError(f"Nonfinite output: {path.name}")
    return array


def encode_text(args, work, weights, t5_storage):
    from transformers import CLIPTokenizer, T5Tokenizer

    clip = CLIPTokenizer.from_pretrained(args.clip_tokenizer, local_files_only=True)
    t5 = T5Tokenizer.from_pretrained(args.t5_tokenizer, local_files_only=True)
    clip_ids = clip(args.prompt, padding="max_length", max_length=77, truncation=True)["input_ids"]
    t5_ids = t5(args.prompt, padding="max_length", max_length=args.text_tokens, truncation=True)["input_ids"]
    clip_output = work / "clip.f32"
    clip_command = [probe_path(args.bin_dir, "clip", "clip_text_probe"), args.backend,
                    weights / "clip.safetensors", clip_output, *map(str, clip_ids)]
    run_probe(clip_command, args.backend, 1, args.expected_device, work)
    clip_hidden = read_tensor(clip_output, (77, 768))
    pooled = clip_hidden[clip_ids.index(clip.eos_token_id)]
    t5_input, t5_output = work / "t5-in.f32", work / "t5-out.f32"
    # FLUX's T5 conditioner passes no attention mask, including for padded tokens.
    np.stack((np.asarray(t5_ids, dtype="<f4"), np.ones(len(t5_ids), dtype="<f4"))).tofile(t5_input)
    t5_command = [probe_path(args.bin_dir, "t5", "t5_encoder"), "t5_xxl", f"--storage={t5_storage}", args.backend,
                  weights / "t5.safetensors", str(len(t5_ids)), t5_input, t5_output]
    run_probe(t5_command, args.backend, 2, args.expected_device, work)
    return read_tensor(t5_output, (len(t5_ids), 4096)), pooled


def denoise(probe, backend, device, profile, storage, weights, image, text, pooled, times, work, adapters=()):
    rows, cols, channels = image.shape
    source, target = work / "flux-in.f32", work / "flux-out.f32"
    with source.open("wb") as stream:
        for values in (image, text, pooled, times):
            np.asarray(values, dtype="<f4", order="C").tofile(stream)
    count = len(times) - 1
    placement = run_probe([probe, profile, storage, backend, weights, str(rows), str(cols),
                           str(len(text)), str(count), source, target, *adapters], backend, max(1, count), device, work)
    return read_tensor(target, (rows, cols, channels)), placement


def unpack(patches):
    rows, cols, _ = patches.shape
    return patches.reshape(rows, cols, 16, 2, 2).transpose(2, 0, 3, 1, 4).reshape(16, rows * 2, cols * 2)


def decode(probe, backend, device, weights, latent, work, *, full=True):
    _, height, width = latent.shape
    source, target = work / "vae-in.f32", work / "vae-out.f32"
    np.asarray(latent, dtype="<f4", order="C").tofile(source)
    mode = ["vae"] if full else []
    run_probe([probe, *mode, backend, weights, str(height), str(width), source, target], backend, 1, device, work)
    return read_tensor(target, (3, height * 8, width * 8))


def generate(args, work):
    weights = args.weights.resolve(strict=True)
    info = json.loads((weights / "flux.json").read_text(encoding="utf-8"))
    if info["profile"] != "schnell" or info["storage"] not in ("f32", "f16", "e4m3", SCALED_FP8) or info["t5_storage"] not in ("f32", "f16"):
        raise ValueError("Expected a compatible Schnell weight bundle")
    probe = probe_path(args.bin_dir, "flux1", "flux1_probe")
    adapter_paths, adapter_report = [], []
    if args.lora:
        import torch

        torch.set_num_threads(4)
        schema = schema_for(probe, "schnell", info["storage"])
        adapter_paths, adapter_report = prepare_loras([(file, float(strength)) for file, strength in args.lora], schema, work)
    text, pooled = encode_text(args, work, weights, info["t5_storage"])
    rng = np.random.default_rng(args.seed)
    image = rng.standard_normal((args.height // 16, args.width // 16, 64), dtype=np.float32)
    times = np.linspace(1, 0, args.steps + 1, dtype="<f4")
    patches, placement = denoise(probe, args.backend, args.expected_device, "schnell", info["storage"],
                                 weights / "flux.safetensors", image, text, pooled, times, work, adapter_paths)
    full = args.decoder == "vae" or (args.decoder == "auto" and (weights / "vae.safetensors").is_file())
    decoder = "vae" if full else "taef1"
    path = probe_path(args.bin_dir, "flux1", "flux1_probe") if full else probe_path(args.bin_dir, "taesd", "taef1_probe")
    rgb = decode(path, args.backend, args.expected_device, weights / (decoder + ".safetensors"), unpack(patches), work, full=full)
    if full:
        rgb = rgb * np.float32(0.5) + np.float32(0.5)
    pixels = np.rint(np.clip(rgb.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
    with args.output.open("xb") as output:
        try:
            Image.fromarray(pixels).save(output, format="PNG")
        except BaseException:
            output.close()
            args.output.unlink()
            raise
    result = {"file": str(args.output.resolve()), "width": args.width, "height": args.height, "seed": args.seed,
              "steps": args.steps, "decoder": decoder, "storage": info["storage"], "placement": placement}
    if adapter_report:
        result["loras"] = adapter_report
    return result


@contextmanager
def work_directory(parent):
    if parent is None:
        with scratch_directory() as work:
            yield work
        return
    with TemporaryDirectory(prefix="flux-", dir=parent.resolve(strict=True)) as work:
        yield Path(work)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file", type=Path, help="Read the prompt from a UTF-8 text file.")
    for name in ("bin-dir", "weights", "clip-tokenizer", "t5-tokenizer", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--backend", choices=PLATFORMS, default="cuda")
    parser.add_argument("--workspace", type=Path, help="Existing directory for temporary files owned by the caller.")
    parser.add_argument("--decoder", choices=("auto", "vae", "taef1"), default="auto")
    parser.add_argument("--expected-device")
    parser.add_argument("--lora", nargs=2, action="append", default=[], metavar=("FILE", "STRENGTH"),
                        help="Apply a local FLUX.1 denoiser LoRA. Repeat to combine adapters.")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text-tokens", type=int, default=256)
    args = parser.parse_args()
    if any(size < 16 or size > 1024 or size % 16 for size in (args.width, args.height)):
        parser.error("Dimensions must be multiples of 16 between 16 and 1024")
    if not 1 <= args.steps <= 50 or not 2 <= args.text_tokens <= 512 or args.seed < 0:
        parser.error("Use 1 to 50 steps, 2 to 512 text tokens and a nonnegative seed")
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("Choose a new PNG path in an existing folder")
    if args.prompt_file is not None:
        with args.prompt_file.open(encoding="utf-8", newline="") as stream:
            args.prompt = stream.read()
    with redirect_stdout(sys.stderr), work_directory(args.workspace) as work:
        os.environ.update({key: str(work) for key in ("TMP", "TEMP", "TMPDIR")})
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(work / "inductor")
        report = generate(args, work)
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
