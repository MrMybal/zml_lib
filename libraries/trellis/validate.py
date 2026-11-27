"""Validate the TRELLIS occupancy structure decoder against FP32 PyTorch."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file
import torch
from torch.nn import functional as F

from structure_io import FORK_TOOLS, coordinates, read_checkpoint, run, schema_for
from build import run_environment, scratch_directory


def reference(weights, latent):
    def conv(x, name):
        return F.conv3d(x, weights[name + ".weight"], weights[name + ".bias"], padding=1)

    def norm(x, name):
        pixels = x.movedim(1, -1)
        return F.layer_norm(pixels, (pixels.shape[-1],), weights[name + ".weight"], weights[name + ".bias"]).movedim(-1, 1)

    def residual(x, name):
        hidden = conv(F.silu(norm(x, name + ".norm1")), name + ".conv1")
        return x + conv(F.silu(norm(hidden, name + ".norm2")), name + ".conv2")

    def expand(x, name):
        projected = conv(x, name)
        n, c8, d, h, w = projected.shape
        # Slice each phase independently to verify the three spatial permutations.
        volume = projected.new_empty((n, c8 // 8, 2 * d, 2 * h, 2 * w))
        for phase in range(8):
            volume[:, :, phase // 4::2, (phase // 2) % 2::2, phase % 2::2] = projected[:, phase::8]
        return volume

    x = conv(latent, "input_layer")
    for block in ("middle_block.0", "middle_block.1", "blocks.0", "blocks.1"):
        x = residual(x, block)
    x = expand(x, "blocks.2.conv")
    x = residual(residual(x, "blocks.3"), "blocks.4")
    x = expand(x, "blocks.5.conv")
    x = residual(residual(x, "blocks.6"), "blocks.7")
    return conv(F.silu(norm(x, "out_layer.0")), "out_layer.2")


def make_fixture(schema, rng):
    fixture = {}
    for key, shape in schema.items():
        value = rng.normal(0, 0.03, size=shape).astype(np.float32)
        if (".norm" in key and key.endswith(".weight")) or key == "out_layer.0.weight":
            value += 1
        if key.startswith(("blocks.", "middle_block.")) and ".conv" in key:
            value = value.astype(np.float16)
        fixture[key] = value
    return fixture


@torch.inference_mode()
def validate(args, work):
    official = schema_for(args.probe, "official")
    print(f"TRELLIS official schema: {len(official)} tensors match", flush=True)
    profile = "official" if args.checkpoint else "fixture"
    rng = np.random.default_rng(771)
    if args.checkpoint:
        fixture = read_checkpoint(args.checkpoint, official)
        sizes = [tuple(args.size)]
    else:
        schema = schema_for(args.probe, "fixture")
        source = work / "mixed-precision.safetensors"
        save_file(make_fixture(schema, rng), source)
        fixture = read_checkpoint(source, schema)
        sizes = [(3, 4, 2), (2, 1, 3)]
    weights = work / "structure.safetensors"
    save_file(fixture, weights)
    if args.reference_device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    reference_weights = {name: torch.from_numpy(value).to(args.reference_device) for name, value in fixture.items()}
    results = []
    for index, size in enumerate(sizes):
        latent = rng.normal(size=(1, 8, *size)).astype(np.float32)
        expected = reference(reference_weights, torch.from_numpy(latent).to(args.reference_device)).cpu().numpy()
        actual = run(args.probe, profile, args.backend, weights, latent, work, str(index), repeats=2,
                     expected_device=args.expected_device, expected_pjrt=args.expected_pjrt)
        tolerance = 3e-5 if profile == "fixture" else 7e-4
        np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance, equal_nan=False)
        # The upstream sparse stage selects positive logits in N, D, H, W order.
        predicted_voxels, expected_voxels = coordinates(actual), coordinates(expected)
        np.testing.assert_array_equal(predicted_voxels, expected_voxels)
        if profile == "fixture" and not 0 < len(predicted_voxels) < actual.size:
            raise AssertionError("Fixture must exercise both occupied and empty voxels")
        max_abs = float(np.max(np.abs(actual - expected)))
        rmse = float(np.sqrt(np.mean((actual.astype(np.float64) - expected) ** 2)))
        results.append(dict(size=list(size), occupied_voxels=len(predicted_voxels), max_abs=max_abs, rmse=rmse))
        print(f"TRELLIS {size}: max logit error {max_abs:.8g}, {len(predicted_voxels)} occupied voxels", flush=True)
    return dict(profile=profile, torch_version=torch.__version__, reference_device=args.reference_device,
                backend=args.backend, expected_pjrt=args.expected_pjrt, expected_device=args.expected_device,
                cases=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--backend", choices=("cpu", "cuda", "rocm", "vulkan"), default="cpu")
    parser.add_argument("--size", nargs=3, type=int, default=(4, 4, 4), metavar=("D", "H", "W"))
    parser.add_argument("--reference-device", default="cpu")
    parser.add_argument("--expected-device")
    parser.add_argument("--expected-pjrt")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if any(not 1 <= dim <= 16 for dim in args.size):
        parser.error("--size edges must be between 1 and 16")
    args.probe = args.probe.resolve(strict=True)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    with scratch_directory() as work:
        os.environ.update(run_environment(work, FORK_TOOLS))
        report = validate(args, work)
    if args.report:
        with args.report.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
