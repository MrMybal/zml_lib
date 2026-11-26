"""Run the native Qwen image, VAE and visual-text probes."""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from inputs import unpack_latent, write_denoise_input

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.probe_placement import verify_placement


def edit_prompt(instruction):
    return (
        "<|im_start|>system\nDescribe the requested edit and preserve the other details.<|im_end|>\n"
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        f"{instruction}<|im_end|>\n<|im_start|>assistant\n"
    )


def run(args, probe, mode, weights, extra, source, output, count, *, flags=()):
    low_memory = args.cuda_low_memory and mode.startswith("edit_2511")
    options = ("--cuda-low-memory",) if low_memory else ()
    command = [probe, *options, mode, args.backend, weights, *map(str, extra), source, output, *flags]
    env = None
    if low_memory and (flags_override := os.getenv("ZML_QWEN_DENOISER_XLA_FLAGS")) is not None:
        env = dict(os.environ, XLA_FLAGS=flags_override)
    proc = subprocess.run(command, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode:
        diagnostic = proc.stderr.strip()
        prefix, has_dump, dump = diagnostic.partition("\nerror(zml/Compiler): Pjrt failed to compile the following MLIR")
        if has_dump and prefix:
            status = [line for line in dump.splitlines()[-4:] if line.startswith("error:")]
            diagnostic = "\n".join([prefix, *status])
        print(diagnostic if len(diagnostic) <= 5000 else diagnostic[:3000] + "\n…\n" + diagnostic[-2000:], file=sys.stderr)
    proc.check_returncode()
    placement = verify_placement(proc.stderr, args.backend, count, args.expected_device, expected_pjrt=args.expected_pjrt)
    if low_memory:
        if "cuda_allocator=bfc" not in proc.stderr or "denoiser_graph=blockwise blocks=60" not in proc.stderr:
            raise RuntimeError("The native Qwen probe did not confirm blockwise BFC execution")
        placement["allocator"] = "bfc"
        placement["graph"] = "blockwise"
        if "weights=streamed" in proc.stderr:
            placement["weights"] = "streamed"
        if "denoiser_attention=tiled" in proc.stderr:
            placement["attention"] = "tiled"
        if os.environ.get("ZML_QWEN_CUDA_MATH"):
            if "qwen_cuda_math=reference" not in proc.stderr:
                raise RuntimeError("The Qwen probe did not confirm the CUDA reference arithmetic")
            placement["math"] = "reference"
    return placement


def vae_run(args, work, profile, weights, value, height, width, label):
    source, output = work / f"{label}.input.f32", work / f"{label}.output.f32"
    np.asarray(value, dtype="<f4", order="C").tofile(source)
    placement = run(args, args.probe, profile, weights, (height, width), source, output, 1)
    shape = (16, height, width) if profile == "vae_encode" else (3, height * 8, width * 8)
    return np.fromfile(output, dtype="<f4").reshape(shape), placement


def run_denoiser(args, work, profile, weights, target, references, text, mask, times, label, *, reference_states=None):
    source, output = work / f"{label}.input.f32", work / f"{label}.output.f32"
    geometry = write_denoise_input(source, target, references, text, mask, times, reference_states=reference_states)
    steps = geometry[-1]
    traced = reference_states is not None
    predictions = 2 * steps - 1 if traced else max(1, steps)
    flags = ("--reference-trajectory",) if traced else ()
    placement = run(args, args.probe, profile, weights, geometry, source, output, predictions, flags=flags)
    rows = np.fromfile(output, dtype="<f4").reshape((steps + 1 if traced else 1), target.size)
    latents = [unpack_latent(row, *target.shape[1:]) for row in rows]
    return (latents if traced else latents[0]), placement


def encode_conditioning(args, work, image, vision_size, prompt):
    from transformers import Qwen2Tokenizer, Qwen2VLImageProcessor
    from libraries.qwen25_vl.inputs import image_positions, text_records, vision_records
    from libraries.qwen25_vl.prepare import prepare as prepare_encoder
    from libraries.qwen25_vl.prepare import read_schema

    tokenizer = Qwen2Tokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    if ids.count(151655) != 1:
        raise ValueError("The conditioning prompt must include exactly one image placeholder")
    start = ids.index(151655)
    pixels = Qwen2VLImageProcessor()(images=image.resize(vision_size), do_resize=False, return_tensors="np")
    grid = pixels.image_grid_thw
    records = vision_records(pixels.pixel_values, grid)
    visual_tokens = len(records) // 4
    if len(ids) - 1 + visual_tokens > 512:
        raise ValueError("The edit prompt and image must fit within 512 Qwen2.5-VL tokens")

    vision = work / "qwen-vl-vision.safetensors"
    prepare_encoder(args.encoder, vision, read_schema(args.encoder_probe, "vision"), "vision")
    source, output = work / "vision.f32", work / "vision-out.f32"
    records.tofile(source)
    run(args, args.encoder_probe, "vision", vision, (len(records),), source, output, 1)
    visual = np.fromfile(output, dtype="<f4").reshape(visual_tokens, 3584)
    vision.unlink()
    expanded = ids[:start] + [151655] * visual_tokens + ids[start + 1:]
    pos = image_positions(start, *(grid[0, 1:] // 2), len(ids) - start - 1)
    packed = text_records(expanded, positions=pos, embeddings=dict(enumerate(visual, start)))
    weights = work / "qwen-vl-text.safetensors"
    profile = "qwen25_vl_7b"
    prepare_encoder(args.encoder, weights, read_schema(args.encoder_probe, profile), "text")
    source, output = work / "text.f32", work / "text-out.f32"
    packed.tofile(source)
    run(args, args.encoder_probe, profile, weights, (len(expanded),), source, output, 1)
    text = np.fromfile(output, dtype="<f4").reshape(len(expanded), 3584)
    weights.unlink()
    return text[start - 1:]
