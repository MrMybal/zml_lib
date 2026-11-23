"""Compare a complete synthetic TRELLIS v1 run with FP32 PyTorch."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from safetensors.numpy import save_file
import torch

from flow_io import FORK_TOOLS, PROFILES, SLAT_MEAN, SLAT_STD, generate_slat, schema_for as flow_schema
from head_io import SCHEMAS as HEADS, decode, schema_for as head_schema
from structure_io import coordinates, read_checkpoint, schema_for as structure_schema
from build import run_environment, scratch_directory
from probe_placement import PLATFORMS
from validate import make_fixture as structure_fixture, reference as reference_structure
from validate_flows import sample_reference
from validate_heads import check_output, reference as reference_head

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dinov2"))
from condition_io import encode, schema_for as dino_schema
from validate_condition import reference as reference_dino


PROFILE = "pipeline-fixture"
STEPS = 25
NOISE_SEED = 371


def synthetic(schema, seed, std=0.035):
    rng = np.random.default_rng(seed)
    weights = {}
    for name, shape in schema.items():
        gain = name.endswith(".gamma") or (name.endswith(".weight") and len(shape) == 1)
        weights[name] = rng.normal(int(gain), std, shape).astype(np.float16)
    return weights


def checkpoint(work, name, schema, weights):
    source, target = work / f"{name}-source.safetensors", work / f"{name}.safetensors"
    save_file(weights, source)
    promoted = read_checkpoint(source, schema)
    save_file(promoted, target)
    return target, promoted


def compare(errors, name, actual, expected, tolerance=2e-5):
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance, equal_nan=False, err_msg=name)
    errors[name] = float(np.max(np.abs(actual - expected)))
    print(f"{name}: shape={actual.shape}, max_abs={errors[name]:.8g}", flush=True)


@torch.inference_mode()
def validate(args, work):
    libraries = args.bin.resolve(strict=True) / "libraries"
    encoder = libraries / "dinov2" / "dinov2_condition.exe"
    trellis = libraries / "trellis"
    flow, structure, heads = (trellis / f"trellis_{stage}.exe" for stage in ("flow", "structure", "heads"))
    device = args.expected_device
    backend = args.backend
    errors = {}

    schema = dino_schema(encoder, "fixture")
    dino_weights, dino_state = checkpoint(work, "dino", schema, synthetic(schema, 23))
    image = np.random.default_rng(81).random((1, 3, 42, 84), dtype=np.float32)
    context = encode(encoder, "fixture", backend, dino_weights, image, work, expected_device=device)
    tokens = work / "condition.npy"
    np.save(tokens, context, allow_pickle=False)
    context = np.load(tokens, allow_pickle=False)
    expected_tokens = reference_dino(dino_state, image, "fixture")
    compare(errors, "dinov2", context, expected_tokens)

    schema = flow_schema(flow, PROFILE, "structure")
    structure_weights, structure_state = checkpoint(work, "structure-flow", schema, synthetic(schema, 2291, 0.07))
    schema = structure_schema(structure, "fixture")
    occupancy_weights, occupancy_state = checkpoint(work, "structure-decoder", schema,
                                                     structure_fixture(schema, np.random.default_rng(771)))
    schema = flow_schema(flow, PROFILE, "slat")
    slat_weights, slat_state = checkpoint(work, "slat-flow", schema, synthetic(schema, 2292, 0.07))
    result = generate_slat(flow, structure, PROFILE, backend, context, work, structure_weights=structure_weights,
                           decoder_weights=occupancy_weights, slat_weights=slat_weights, steps=STEPS,
                           rng=np.random.default_rng(NOISE_SEED), expected_device=device)

    rng = np.random.default_rng(NOISE_SEED)
    zeros = np.zeros_like(expected_tokens)
    _, grid = PROFILES[PROFILE]
    noise = rng.standard_normal((1, 8, grid, grid, grid), dtype=np.float32)
    dense = sample_reference(structure_state, noise, expected_tokens, zeros, steps=STEPS)
    compare(errors, "structure_flow", result.structure, dense)
    weights = {key: torch.from_numpy(value) for key, value in occupancy_state.items()}
    logits = reference_structure(weights, torch.from_numpy(dense)).numpy()
    compare(errors, "occupancy_logits", result.logits, logits)
    expected_coords = coordinates(logits)
    np.testing.assert_array_equal(result.coordinates, expected_coords)
    if not 0 < len(expected_coords) < logits.size:
        raise AssertionError("The pipeline fixture must contain occupied and empty voxels")

    noise = rng.standard_normal((len(expected_coords), 8), dtype=np.float32)
    features = sample_reference(slat_state, noise, expected_tokens, zeros, coords=expected_coords, steps=STEPS)
    compare(errors, "slat_flow", result.sampled_features, features)
    expected_features = features * SLAT_STD + SLAT_MEAN
    slat = work / "slat.npz"
    np.savez(slat, coordinates=result.coordinates, features=result.features)
    with np.load(slat, allow_pickle=False) as archive:
        cells, features = archive["coordinates"], archive["features"]
    np.testing.assert_array_equal(cells, expected_coords)
    compare(errors, "slat_denormalized", features, expected_features)

    exports = {}
    for index, kind in enumerate(HEADS):
        schema = head_schema(heads, "fixture", kind)
        state = synthetic(schema, 91 + index, 0.045)
        if kind == "mesh":
            state["out_layer.bias"][:8] = -0.16
        weights, state = checkpoint(work, kind, schema, state)
        xyz, values = decode(heads, "fixture", kind, backend, weights, cells, features, work, expected_device=device)
        expected_xyz, expected_values = reference_head(state, expected_coords, expected_features, kind)
        np.testing.assert_array_equal(xyz, expected_xyz)
        compare(errors, kind, values, expected_values)
        output = work / (f"{kind}.ply" if kind == "gaussian" else f"{kind}.npz")
        exports[kind] = check_output(kind, output, xyz, values, expected_values)

    print(json.dumps({"euler_steps_per_flow": STEPS, "occupied_voxels": len(cells),
                      "max_abs_errors": errors, "exports": exports}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", required=True, type=Path)
    parser.add_argument("--backend", choices=PLATFORMS, default="cpu")
    parser.add_argument("--expected-device", help="substring of the reported output device name")
    args = parser.parse_args()
    torch.set_num_threads(2)
    with scratch_directory() as work:
        os.environ.update(run_environment(work, FORK_TOOLS))
        if args.backend == "cuda":
            os.environ["CUDA_CACHE_DISABLE"] = "1"
        validate(args, work)


if __name__ == "__main__":
    main()
