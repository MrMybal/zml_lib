"""Repair a masked region using the local nine-channel SD 1.5 model in ZML."""

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image

from generate import (add_options, check_options, decode, encode_prompts, image_pixels,
                      predict, read_tensor, run, run_probe, write_png)
from prepare import probe_path


PROFILE = "sd15-inpaint"


def add_inpaint_options(parser, *, weights_required=True, inputs_required=True):
    add_options(parser, weights_required=weights_required, size_options=False)
    parser.add_argument("--image", required=inputs_required, type=Path)
    parser.add_argument("--mask", required=inputs_required, type=Path)
    parser.add_argument("--strength", type=float, default=1.0)


def check_inputs(args, parser):
    with Image.open(args.image) as source, Image.open(args.mask) as mask:
        if source.size != mask.size:
            parser.error("Image and mask dimensions must match.")
        args.width, args.height = source.size
        check_options(args, parser, PROFILE)
        if not math.isfinite(args.strength) or not 0 < args.strength <= 1:
            parser.error("Strength must be above zero and at most 1.")
        args.effective_steps = int(args.steps * args.strength)
        if args.effective_steps == 0:
            parser.error("Increase strength or steps to get at least one DDIM step.")
        return np.array(source.convert("RGB")), np.array(mask.convert("L")) >= 128


def create_noise(args):
    rng = np.random.default_rng(args.seed)
    shape = (4, args.height // 8, args.width // 8)
    return [rng.standard_normal(shape, dtype=np.float32) for _ in range(2 + int(args.effective_steps < args.steps))]


def encode_image(args, work, rgb, mask, posterior_noise):
    graph = probe_path(args.bin_dir, "stable_diffusion", "sd15_probe")
    source, target = work / "encoder-in.f32", work / "encoder-out.f32"
    with source.open("wb") as stream:
        (rgb.transpose(2, 0, 1).astype("<f4") / 127.5 - 1).tofile(stream)
        mask.astype("<f4").tofile(stream)
        for noise in posterior_noise:
            noise.tofile(stream)
    count = len(posterior_noise)
    run_probe([graph, "vae_encode", args.backend, args.weights / "encoder.safetensors",
               str(args.height // 8), str(args.width // 8), "both" if count == 2 else "masked", source, target],
              args.backend, count, args.expected_device, work)
    return read_tensor(target, (count, 4, args.height // 8, args.width // 8))


def composite(rgb, mask, decoded):
    return np.where(mask[:, :, None], image_pixels(decoded), rgb)


def inpaint(args, work, rgb, mask):
    positive, negative = encode_prompts(args, work)
    initial, *posterior_noise = create_noise(args)
    masked, *source = encode_image(args, work, rgb, mask, posterior_noise)
    condition = (mask[None, ::8, ::8], masked)
    start = args.steps - args.effective_steps
    latent = predict(args, work, initial, positive, negative=negative, image_condition=condition,
                     source_latent=source[0] if source else None, start=start)
    result, placement = decode(args, work, latent)
    write_png(args.output, composite(rgb, mask, result))
    return {"file": str(args.output), "width": args.width, "height": args.height, "steps": args.steps,
            "seed": args.seed, "guidance": args.guidance, "strength": args.strength,
            "effective_steps": args.effective_steps, "scheduler": "ddim", "placement": placement}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_inpaint_options(parser)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rgb, mask = check_inputs(args, parser)
    run(args, parser, lambda args, work: inpaint(args, work, rgb, mask))


if __name__ == "__main__":
    main()
