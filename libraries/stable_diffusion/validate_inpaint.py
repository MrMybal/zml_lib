"""Compare native SD 1.5 Inpainting stages against Diffusers and PyTorch FP32."""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
from PIL import Image

from generate import decode, encode_prompts, image_pixels, predict, scratch_directory, token_ids
from inpaint import PROFILE, add_inpaint_options, check_inputs, composite, create_noise, encode_image
from inpaint_fixture import write_fixture
from prepare import prepare
from validate import clip_reference, compare, ddim_reference, load_unet, load_vae, numpy, set_validation_caches


def reference(args, initial, posterior_noise):
    import torch
    from diffusers.image_processor import VaeImageProcessor
    from transformers import CLIPTokenizer

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = args.torch_device
    tokenizer = CLIPTokenizer.from_pretrained(args.clip_tokenizer, local_files_only=True)
    ids = token_ids(tokenizer, args.prompt)
    negative_ids = token_ids(tokenizer, args.negative_prompt) if args.guidance > 1 else None
    positive, negative = clip_reference(args, ids, negative_ids)
    processor = VaeImageProcessor(do_convert_rgb=True)
    mask_processor = VaeImageProcessor(do_normalize=False, do_binarize=True, do_convert_grayscale=True)
    with Image.open(args.image) as image, Image.open(args.mask) as image_mask:
        original = processor.preprocess(image, height=args.height, width=args.width).to(device)
        mask = mask_processor.preprocess(image_mask, height=args.height, width=args.width).to(device)
    vae = load_vae(args)

    def sample(image, noise):
        posterior = vae.encode(image).latent_dist
        return (posterior.mean + posterior.std * torch.from_numpy(noise[None]).to(device)) * vae.config.scaling_factor

    masked = sample(original * (mask < 0.5), posterior_noise[0])
    source = sample(original, posterior_noise[1]) if len(posterior_noise) == 2 else None
    mask_latent = torch.nn.functional.interpolate(mask, size=masked.shape[-2:], mode="nearest")
    scheduler = ddim_reference(args)
    timesteps = scheduler.timesteps[args.steps - args.effective_steps:]
    latent = torch.from_numpy(initial[None]).to(device)
    if source is not None:
        latent = scheduler.add_noise(source, latent, timesteps[:1])
    first_latent = numpy(latent)
    unet = load_unet(args, input_channels=9)
    first = None
    for timestep in timesteps:
        model_input = torch.cat((scheduler.scale_model_input(latent, timestep), mask_latent, masked), dim=1)
        prediction = unet(model_input, timestep, positive).sample
        if first is None:
            first = numpy(prediction)
        if negative is not None:
            unconditional = unet(model_input, timestep, negative).sample
            prediction = unconditional + args.guidance * (prediction - unconditional)
        latent = scheduler.step(prediction, timestep, latent, eta=0).prev_sample
    del unet
    rgb = vae.decode(latent / vae.config.scaling_factor).sample
    return {"positive": numpy(positive), "negative": numpy(negative) if negative is not None else None,
            "mask": numpy(mask_latent), "masked": numpy(masked), "source": numpy(source) if source is not None else None,
            "first_latent": first_latent, "timestep": int(timesteps[0]), "prediction": first,
            "latent": numpy(latent), "rgb": numpy(rgb), "pixels": np.array(processor.postprocess(rgb)[0])}


def validate(args, work, rgb, mask):
    import torch

    initial, *posterior_noise = create_noise(args)
    with torch.inference_mode():
        expected = reference(args, initial, posterior_noise)
    gc.collect()
    if args.torch_device.startswith("cuda"):
        torch.cuda.empty_cache()
    positive, negative = encode_prompts(args, work)
    masked, *source = encode_image(args, work, rgb, mask, posterior_noise)
    mask_latent = mask[None, ::8, ::8]
    predicted = predict(args, work, expected["first_latent"], expected["positive"], timestep=expected["timestep"],
                        image_condition=(expected["mask"], expected["masked"]))
    latent = predict(args, work, initial, positive, negative=negative, image_condition=(mask_latent, masked),
                     source_latent=source[0] if source else None, start=args.steps - args.effective_steps)
    same_latent, placement = decode(args, work, expected["latent"], "vae-reference-latent")
    generated, _ = decode(args, work, latent)
    output = composite(rgb, mask, generated)
    expected_output = np.where(mask[:, :, None], expected["pixels"], rgb)
    results = {
        "clip_positive": compare(positive, expected["positive"], max_limit=0.003, mean_limit=0.0001),
        "mask_latent": compare(mask_latent, expected["mask"], max_limit=0, mean_limit=0),
        "masked_image_latent": compare(masked, expected["masked"], max_limit=0.003, mean_limit=0.0003),
        "unet_prediction": compare(predicted, expected["prediction"], max_limit=0.003, mean_limit=0.0003),
        "ddim_latent": compare(latent, expected["latent"], max_limit=0.02, mean_limit=0.002),
        "vae_decoder": compare(same_latent, expected["rgb"], max_limit=0.006, mean_limit=0.0006),
        "raw_rgb8": compare(image_pixels(generated), expected["pixels"], max_limit=8, mean_limit=0.5),
        "composite_rgb8": compare(output, expected_output, max_limit=8, mean_limit=0.5),
    }
    if source:
        results["source_image_latent"] = compare(source[0], expected["source"], max_limit=0.003, mean_limit=0.0003)
    if negative is not None:
        results["clip_negative"] = compare(negative, expected["negative"], max_limit=0.003, mean_limit=0.0001)
    changed_inside = int(np.count_nonzero(np.any(output != rgb, axis=2) & mask))
    changed_outside = int(np.count_nonzero(np.any(output != rgb, axis=2) & ~mask))
    report = {"placement": placement, "strength": args.strength, "steps": args.effective_steps,
              "width": args.width, "height": args.height, "changed_inside": changed_inside,
              "changed_outside": changed_outside, "comparisons": results}
    print(json.dumps(report, indent=2))
    if changed_outside or (mask.any() and not changed_inside) or not all(result["passed"] for result in results.values()):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_inpaint_options(parser, weights_required=False, inputs_required=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--torch-device", default="cuda:0")
    parser.set_defaults(steps=4, strength=0.75)
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve(strict=True)
    if (args.image is None) != (args.mask is None):
        parser.error("Supply both --image and --mask, or omit both to use the 192 × 128 fixture.")
    with scratch_directory() as work:
        set_validation_caches(work)
        if args.image is None:
            args.image, args.mask = write_fixture(work, 192, 128)
        if args.weights is None:
            args.weights = work / "weights"
            prepare(args.checkpoint, args.bin_dir, args.weights, PROFILE)
        image, mask = check_inputs(args, parser)
        validate(args, work, image, mask)


if __name__ == "__main__":
    main()
