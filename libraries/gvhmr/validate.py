"""Compare the SIGA24 port with the installed GVHMR PyTorch model."""

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import torch

from temporal_io import CONDITION_SHAPES, FORK_TOOLS, load_checkpoint, read_conditions, run, save_weights, schema_for
from build import run_environment, scratch_directory


def reference_for(upstream, weights, schema):
    sys.path.insert(0, str(upstream.resolve(strict=True)))
    from hmr4d.network.gvhmr.relative_transformer import NetworkEncoderRoPE

    model = NetworkEncoderRoPE().eval()
    expected_schema = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
    if expected_schema != schema:
        raise ValueError("The Zig schema differs from the pinned GVHMR reference")
    model.load_state_dict(weights, strict=True)
    return model


def sample(frames, rng):
    condition = {
        name: rng.normal(0, 0.5, size=(frames, *shape)).astype(np.float32)
        for name, shape in CONDITION_SHAPES.items()
    }
    condition["obs"][..., 2] = rng.uniform(0.04, 0.99, size=(frames, 17))
    condition["obs"][:, :3, 2] = np.asarray([0.2, 0.5, 0.9], dtype=np.float32)
    return condition


@torch.inference_mode()
def validate(args, work):
    schema = schema_for(args.probe)
    weights = load_checkpoint(args.checkpoint, schema)
    reference = reference_for(args.upstream, weights, schema)
    print(f"GVHMR SIGA24: {len(schema)} official FP32 tensors match the reference", flush=True)
    converted = work / "temporal.safetensors"
    save_weights(weights, converted)
    rng = np.random.default_rng(1372)
    clips = [read_conditions(args.conditions)] if args.conditions else [sample(n, rng) for n in (7, 137)]
    for i, clip in enumerate(clips):
        frames = len(clip["obs"])
        native = run(args.probe, args.backend, converted, clip, work, str(i), repeats=2)
        values = {name: torch.from_numpy(np.asarray(value, dtype=np.float32))[None] for name, value in clip.items()}
        expected = reference(length=torch.tensor([frames]), **values)
        for name, actual in native.items():
            predicted = expected[name][0].numpy()
            np.testing.assert_allclose(actual, predicted, atol=2e-4, rtol=2e-4, equal_nan=False)
            print(f"GVHMR {frames} frames {name}: max absolute error {np.max(np.abs(actual - predicted)):.8g}", flush=True)
        shape_coefficients = native["pred_x"][:, 126:136]
        np.testing.assert_array_equal(shape_coefficients, np.broadcast_to(shape_coefficients[0], shape_coefficients.shape))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--conditions", type=Path)
    parser.add_argument("--backend", choices=("cpu", "cuda", "rocm", "vulkan"), default="cpu")
    args = parser.parse_args()
    args.probe = args.probe.resolve(strict=True)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    with scratch_directory() as work:
        os.environ.update(run_environment(work, FORK_TOOLS))
        validate(args, work)


if __name__ == "__main__":
    main()
