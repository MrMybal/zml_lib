"""Validate the two TRELLIS rectified flows against FP32 PyTorch."""

import argparse
import os
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file
import torch
from torch.nn import functional as F

from flow_io import FORK_TOOLS, sample, schedule, schema_for
from structure_io import read_checkpoint
from build import run_environment, scratch_directory


def reference(weights, state, context, time, xyz, heads):
    def linear(x, key):
        return F.linear(x, weights[key + ".weight"], weights[key + ".bias"])

    def layer(x, eps, key=None):
        return F.layer_norm(x, (x.shape[-1],), weights.get(f"{key}.weight"), weights.get(f"{key}.bias"), eps)

    def frequency(n):
        return 10000 ** (-torch.arange(n, dtype=torch.float32) / n)

    def spatial_conv(x, loc, key):
        # Independent dense cross-correlation, evaluated only at active voxels.
        dims = tuple((loc.max(0).values + 1).tolist())
        grid = torch.zeros((1, x.shape[1], *dims), dtype=x.dtype)
        d, h, w = loc.T
        grid[0, :, d, h, w] = x.T
        kernel = weights[key + ".weight"].permute(0, 4, 1, 2, 3)
        volume = F.conv3d(grid, kernel, weights[key + ".bias"], padding=1)
        return volume[0, :, d, h, w].T

    angle = time * frequency(128)
    embedding = linear(F.silu(linear(torch.cat((angle.cos(), angle.sin()))[None], "t_embedder.mlp.0")), "t_embedder.mlp.2")

    def spatial(x, location, prefix):
        inner = spatial_conv(F.silu(layer(x, 1e-6, prefix + ".norm1")), location, prefix + ".conv1.conv")
        scale, shift = linear(F.silu(embedding), prefix + ".emb_layers.1").chunk(2, dim=-1)
        y = spatial_conv(F.silu(layer(inner, 1e-6) * (1 + scale) + shift), location, prefix + ".conv2.conv")
        skip = prefix + ".skip_connection"
        return y + (linear(x, skip) if skip + ".weight" in weights else x)

    hidden = linear(state, "input_layer")
    if xyz is None:
        tokens = hidden + weights["pos_emb"]
    else:
        local = spatial(hidden, xyz, "input_blocks.0")
        grid, parent = torch.unique(xyz // 2, dim=0, sorted=True, return_inverse=True)
        pool = torch.zeros((len(grid), local.shape[1]), dtype=local.dtype).scatter_reduce(
            0, parent[:, None].expand_as(local), local, reduce="mean",
        )
        down = spatial(pool, grid, "input_blocks.1")
        phases = grid[..., None].float() * frequency(down.shape[-1] // 6)
        pos = torch.cat((phases.sin(), phases.cos()), dim=-1).flatten(1)
        tokens = down + F.pad(pos, (0, down.shape[-1] - pos.shape[-1]))

    def split(x):
        return x.unflatten(-1, (heads, -1)).transpose(0, 1)

    def attend(q, k, v):
        return F.scaled_dot_product_attention(q, k, v).transpose(0, 1).flatten(1)

    blocks = sum(key.endswith("adaLN_modulation.1.bias") for key in weights)
    for i in range(blocks):
        prefix = f"blocks.{i}."
        a, b, c, d, e, f = linear(F.silu(embedding), prefix + "adaLN_modulation.1").chunk(6, dim=-1)
        y = layer(tokens, 1e-6) * (1 + b) + a
        q, k, v = (split(group) for group in linear(y, prefix + "self_attn.to_qkv").chunk(3, dim=-1))
        key = prefix + "self_attn."
        q = F.normalize(q, dim=-1) * weights[key + "q_rms_norm.gamma"][:, None, :] * q.shape[-1] ** 0.5
        k = F.normalize(k, dim=-1) * weights[key + "k_rms_norm.gamma"][:, None, :] * k.shape[-1] ** 0.5
        tokens = tokens + c * linear(attend(q, k, v), key + "to_out")
        key = prefix + "cross_attn."
        q = split(linear(layer(tokens, 1e-6, prefix + "norm2"), key + "to_q"))
        k, v = (split(group) for group in linear(context, key + "to_kv").chunk(2, dim=-1))
        tokens = tokens + linear(attend(q, k, v), key + "to_out")
        y = linear(layer(tokens, 1e-6) * (1 + e) + d, prefix + "mlp.mlp.0")
        tokens = tokens + f * linear(F.gelu(y, approximate="tanh"), prefix + "mlp.mlp.2")

    if xyz is not None:
        expand = torch.cat((tokens, down), dim=-1)[parent]
        out = spatial(expand, xyz, "out_blocks.0")
        tokens = spatial(torch.cat((out, local), dim=-1), xyz, "out_blocks.1")
    return linear(layer(tokens, 1e-5), "out_layer")


def sample_reference(params, noise, context, negative, *, coords=None, heads=2, steps=25, guidance=5.0, rescale=3.0):
    xyz = None if coords is None else torch.from_numpy(coords[:, 1:]).long()
    data = torch.from_numpy(noise[0].reshape(8, -1).T.copy() if xyz is None else noise.copy())
    positive, negative = torch.from_numpy(context), torch.from_numpy(negative)
    model = {name: torch.from_numpy(value) for name, value in params.items()}
    times = schedule(steps, rescale)
    for start, end in zip(times[:-1], times[1:]):
        velocity = reference(model, data, positive, 1000 * start, xyz, heads)
        if guidance and start >= 0.5:
            against = reference(model, data, negative, 1000 * start, xyz, heads)
            velocity = (1 + guidance) * velocity - guidance * against
        data -= float(start - end) * velocity
    return data.T.reshape(noise.shape).numpy() if xyz is None else data.numpy()


@torch.inference_mode()
def validate(args, work):
    rng = np.random.default_rng(2291)
    coords = np.array([[0, *p] for p in [
        (2, 2, 2), (0, 0, 0), (0, 1, 0), (1, 0, 1), (2, 1, 2), (7, 7, 7), (4, 2, 3)
    ]], dtype=np.int32)
    choices = {"structure": args.structure_checkpoint, "slat": args.slat_checkpoint}
    for stage, checkpoint in choices.items():
        official = schema_for(args.probe, "official", stage)
        print(f"TRELLIS {stage} official schema: {len(official)} tensors", flush=True)
        profile = "official" if checkpoint else "fixture"
        if checkpoint:
            params = read_checkpoint(checkpoint, official)
        else:
            params = {}
            for key, shape in schema_for(args.probe, profile, stage).items():
                value = rng.normal(0, 0.07, size=shape).astype(np.float32)
                if key.endswith(".gamma") or (".norm" in key and key.endswith(".weight")):
                    value += 1
                params[key] = value
        weights = work / f"{stage}.safetensors"
        save_file(params, weights)
        heads, width, grid = (16, 1024, 16) if checkpoint else (2, 12, 3)
        occupied = None if stage == "structure" else coords
        noise = rng.normal(size=(1, 8, grid, grid, grid) if occupied is None else (len(coords), 8)).astype(np.float32)
        context = rng.normal(size=(5, width)).astype(np.float32)
        other = rng.normal(size=context.shape).astype(np.float32)
        steps = 4 if not checkpoint else args.steps
        expected = sample_reference(params, noise, context, other, coords=occupied, heads=heads,
                                    steps=steps, guidance=3, rescale=2)
        native = sample(args.probe, args.backend, profile, stage, weights, noise, context, other, work,
                        coords=occupied, steps=steps, guidance=3, rescale=2)
        tolerance = 2e-4 if not checkpoint else 2e-3
        np.testing.assert_allclose(native, expected, atol=tolerance, rtol=tolerance, equal_nan=False)
        print(f"TRELLIS {stage}, {steps} Euler steps: max absolute error {np.abs(native-expected).max():.8g}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--structure-checkpoint", type=Path)
    parser.add_argument("--slat-checkpoint", type=Path)
    parser.add_argument("--backend", choices=("cpu", "cuda", "rocm", "vulkan"), default="cpu")
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    args.probe = args.probe.resolve(strict=True)
    if not 1 <= args.steps <= 1000:
        parser.error("--steps must be 1–1000")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    with scratch_directory() as work:
        os.environ.update(run_environment(work, FORK_TOOLS))
        validate(args, work)


if __name__ == "__main__":
    main()
