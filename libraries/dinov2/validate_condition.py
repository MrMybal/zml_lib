"""Compare native DINOv2 conditioning on a forced backend with CPU PyTorch."""

import argparse
import os
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file
import torch
from torch.nn import functional as F

from condition_io import BACKENDS, PROFILES, TOOLS, encode, read_checkpoint, schema_for
from build import run_environment, scratch_directory


def reference(state, image, profile):
    width, grid, layers, heads = PROFILES[profile]
    w = {name.removeprefix("pretrained."): torch.from_numpy(value) for name, value in state.items()}
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    x = (torch.from_numpy(image) - mean) / std
    patches = F.conv2d(x, w["patch_embed.proj.weight"], w["patch_embed.proj.bias"], stride=14)
    position = w["pos_embed"]
    if patches.shape[2:] != (grid, grid):
        grid_pos = position[:, 1:].reshape(1, grid, grid, width).permute(0, 3, 1, 2)
        grid_pos = F.interpolate(grid_pos, size=patches.shape[2:], mode="bicubic", align_corners=False, antialias=True)
        position = torch.cat((position[:, :1], grid_pos.flatten(2).transpose(1, 2)), dim=1)
    tokens = torch.cat((w["cls_token"], patches.flatten(2).transpose(1, 2)), dim=1) + position
    x = torch.cat((tokens[:, :1], w["register_tokens"], tokens[:, 1:]), dim=1)

    def linear(value, name):
        return F.linear(value, w[name + ".weight"], w[name + ".bias"])

    def norm(value, name):
        return F.layer_norm(value, (width,), w[name + ".weight"], w[name + ".bias"], eps=1e-6)

    for i in range(layers):
        prefix = f"blocks.{i}."
        qkv = linear(norm(x, prefix + "norm1"), prefix + "attn.qkv")
        q, k, v = qkv.reshape(1, x.shape[1], 3, heads, width // heads).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape_as(x)
        x = x + linear(attended, prefix + "attn.proj") * w[prefix + "ls1.gamma"]
        hidden = F.gelu(linear(norm(x, prefix + "norm2"), prefix + "mlp.fc1"))
        x = x + linear(hidden, prefix + "mlp.fc2") * w[prefix + "ls2.gamma"]
    return F.layer_norm(x, (width,)).squeeze(0).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--backend", choices=BACKENDS, default="cpu")
    parser.add_argument("--expected-device", help="substring of the reported output device name")
    args = parser.parse_args()
    torch.set_num_threads(2)
    rng = np.random.default_rng(23)
    probe = args.probe.resolve(strict=True)
    official = schema_for(probe, "registered")
    profile = "registered" if args.checkpoint else "fixture"
    schema = official if args.checkpoint else schema_for(probe, profile)
    converted = read_checkpoint(args.checkpoint, schema) if args.checkpoint else None
    sizes = ((518, 518),) if args.checkpoint else ((42, 84), (70, 70))
    for h, w in sizes:
        image = rng.random((1, 3, h, w), dtype=np.float32)
        with scratch_directory() as work:
            os.environ.update(run_environment(work, TOOLS))
            weights = work / "dino.safetensors"
            state = converted
            if state is None:
                source = work / "dino.pth"
                torch.save({name.removeprefix("pretrained."): torch.from_numpy(rng.normal(
                    1 if "norm" in name and name.endswith(".weight") else 0, 0.035, dims).astype(np.float16))
                    for name, dims in schema.items()}, source)
                state = read_checkpoint(source, schema)
            save_file(state, weights)
            actual = encode(probe, profile, args.backend, weights, image, work, repeats=2, expected_device=args.expected_device)
            expected = reference(state, image, profile)
            tolerance = 5e-4 if args.checkpoint else 1e-5
            np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
            print(f"DINOv2 {h}x{w}, shape={actual.shape}, max_abs={np.max(np.abs(actual - expected)):.8g}", flush=True)


if __name__ == "__main__":
    main()
