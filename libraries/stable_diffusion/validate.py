"""Compare the ZML SD 1.5 pipeline with PyTorch from the same original checkpoint."""

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np

from generate import (add_options, check_options, decode, encode, image_pixels, initial_noise,
                      predict, scratch_directory, token_ids)
from prepare import clip_config, open_checkpoint, prepare


def set_validation_caches(work):
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(work / "inductor")
    os.environ["TRITON_CACHE_DIR"] = str(work / "triton")


def take(path, prefix):
    with open_checkpoint(path) as (keys, get_tensor):
        return {name: get_tensor(name).float() for name in keys if name.startswith(prefix)}


def clip_reference(args, ids, negative_ids):
    import torch
    from transformers import CLIPTextModel

    device = args.torch_device
    source = take(args.checkpoint, "cond_stage_model.transformer.")
    weights = {key.removeprefix("cond_stage_model.transformer."): value for key, value in source.items()
               if not key.endswith(".position_ids")}
    # CLIP computes its nonpersistent position buffer at initialization.
    clip = CLIPTextModel(clip_config()).eval()
    clip.load_state_dict(weights, strict=True)
    clip.to(device)
    positive = clip(torch.tensor([ids], device=device)).last_hidden_state
    negative = clip(torch.tensor([negative_ids], device=device)).last_hidden_state if negative_ids else None
    return positive, negative


def load_unet(args, *, input_channels=4):
    import torch
    from diffusers import UNet2DConditionModel
    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import convert_ldm_unet_checkpoint

    with torch.device("meta"):
        unet = UNet2DConditionModel(
            sample_size=64, in_channels=input_channels, out_channels=4, layers_per_block=2, cross_attention_dim=768,
            block_out_channels=(320, 640, 1280, 1280), attention_head_dim=8,
            down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        ).eval()
    weights = convert_ldm_unet_checkpoint(take(args.checkpoint, "model.diffusion_model."), unet.config)
    unet.load_state_dict(weights, strict=True, assign=True)
    return unet.to(args.torch_device)


def ddim_reference(args):
    from diffusers import DDIMScheduler

    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                              clip_sample=False, set_alpha_to_one=False, steps_offset=1)
    scheduler.set_timesteps(args.steps, device=args.torch_device)
    return scheduler


def load_vae(args):
    import torch
    from diffusers import AutoencoderKL
    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import convert_ldm_vae_checkpoint

    with torch.device("meta"):
        vae = AutoencoderKL(
            block_out_channels=(128, 256, 512, 512), latent_channels=4, layers_per_block=2,
            down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4,
            scaling_factor=0.18215, sample_size=512,
        ).eval()
    weights = convert_ldm_vae_checkpoint(take(args.checkpoint, "first_stage_model."), vae.config)
    vae.load_state_dict(weights, strict=True, assign=True)
    return vae.to(args.torch_device)


def numpy(value):
    return value.detach().cpu().numpy()[0]


def reference(args, ids, negative_ids, initial):
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    positive, negative = clip_reference(args, ids, negative_ids)
    unet = load_unet(args)
    scheduler = ddim_reference(args)
    latent = torch.from_numpy(initial[None]).to(args.torch_device)
    first = None
    for t in scheduler.timesteps:
        prediction = unet(latent, t, positive).sample
        if first is None:
            first = numpy(prediction)
        if negative is not None:
            unconditional = unet(latent, t, negative).sample
            prediction = unconditional + args.guidance * (prediction - unconditional)
        latent = scheduler.step(prediction, t, latent, eta=0).prev_sample
    del unet
    vae = load_vae(args)
    rgb = vae.decode(latent / vae.config.scaling_factor).sample
    return {"positive": numpy(positive), "negative": numpy(negative) if negative is not None else None,
            "first_timestep": int(scheduler.timesteps[0]), "prediction": first,
            "latent": numpy(latent), "rgb": numpy(rgb)}


def compare(actual, expected, *, max_limit, mean_limit):
    errors = np.abs(actual.astype("float64") - expected)
    maximum, mean = float(errors.max()), float(errors.mean())
    return {"max_abs": maximum, "mean_abs": mean, "passed": maximum <= max_limit and mean <= mean_limit}


def validate(args, work):
    import torch
    from transformers import CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(args.clip_tokenizer, local_files_only=True)
    ids = token_ids(tokenizer, args.prompt)
    negative_ids = token_ids(tokenizer, args.negative_prompt) if args.guidance > 1 else None
    initial = initial_noise(args.seed, args.height, args.width)
    with torch.inference_mode():
        expected = reference(args, ids, negative_ids, initial)
    gc.collect()
    if args.torch_device.startswith("cuda"):
        torch.cuda.empty_cache()
    positive = encode(args, work, ids, "positive")
    negative = encode(args, work, negative_ids, "negative") if negative_ids else None
    prediction = predict(args, work, initial, expected["positive"], timestep=expected["first_timestep"])
    latent = predict(args, work, initial, positive, negative=negative)
    same_latent_rgb, placement = decode(args, work, expected["latent"], "vae-reference-latent")
    end_to_end_rgb, _ = decode(args, work, latent)
    results = {
        "clip_positive": compare(positive, expected["positive"], max_limit=0.003, mean_limit=0.0001),
        "unet": compare(prediction, expected["prediction"], max_limit=0.003, mean_limit=0.0003),
        "ddim_latent": compare(latent, expected["latent"], max_limit=0.02, mean_limit=0.002),
        "vae": compare(same_latent_rgb, expected["rgb"], max_limit=0.006, mean_limit=0.0006),
        "pipeline_rgb8": compare(image_pixels(end_to_end_rgb), image_pixels(expected["rgb"]),
                                 max_limit=8, mean_limit=0.5),
    }
    if negative is not None:
        results["clip_negative"] = compare(negative, expected["negative"], max_limit=0.003, mean_limit=0.0001)
    print(json.dumps({"placement": placement, "comparisons": results}, indent=2))
    if not all(result["passed"] for result in results.values()):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_options(parser, weights_required=False)
    parser.set_defaults(width=128, height=128, steps=4)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--torch-device", default="cuda:0")
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve(strict=True)
    with scratch_directory() as work:
        set_validation_caches(work)
        if args.weights is None:
            args.weights = work / "weights"
            prepare(args.checkpoint, args.bin_dir, args.weights)
        check_options(args, parser)
        validate(args, work)


if __name__ == "__main__":
    main()
