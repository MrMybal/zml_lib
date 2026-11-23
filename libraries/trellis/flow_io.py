"""Run the TRELLIS-image-large structure and SLAT flows with local weights."""

import argparse
from hashlib import sha256
import os
from pathlib import Path
import subprocess
import sys
from typing import NamedTuple

import numpy as np
from safetensors.numpy import save_file

from structure_io import FORK_TOOLS, coordinates, read_checkpoint, run as decode_structure
from build import run_environment, scratch_directory
from probe_placement import PLATFORMS, verify_placement


SCHEMAS = {
    "structure": "72a3d8bcbcfa16dd35787b38f7c9aa51cae06e40cdf0d43467fc83ae948387ec",
    "slat": "3f731c6313c2e1f50a358a8708ffab74c9a433c604b16f017fe18d9c9c9dcf04",
}
PROFILES = {"official": (1024, 16), "fixture": (12, 3), "pipeline-fixture": (24, 2)}
SLAT_MEAN = np.array([-2.1687545776367188, -0.004347046371549368, -0.13352349400520325, -0.08418072760105133,
                     -0.5271206498146057, 0.7238689064979553, -1.1414450407028198, 1.2039363384246826], dtype=np.float32)
SLAT_STD = np.array([2.377650737762451, 2.386378288269043, 2.124418020248413, 2.1748552322387695,
                    2.663944721221924, 2.371192216873169, 2.6217446327209473, 2.684523105621338], dtype=np.float32)
OFFSETS = np.indices((3, 3, 3), dtype=np.int32).reshape(3, -1).T - 1


def schema_for(probe, profile, stage):
    schema = {}
    lines = subprocess.check_output([probe, "schema", profile, stage], text=True).splitlines()
    for line in lines:
        name, *dims = line.split("\t")
        if name in schema:
            raise ValueError(f"Duplicate flow tensor {name}")
        schema[name] = tuple(map(int, dims))
    if profile == "official":
        manifest = "\n".join(name + ":" + ",".join(map(str, shape)) for name, shape in sorted(schema.items()))
        if sha256(manifest.encode("ascii")).hexdigest() != SCHEMAS[stage]:
            raise ValueError(f"{stage} schema differs from the official checkpoint")
    return schema


def neighbors(xyz, edge):
    table = np.full((edge, edge, edge), len(xyz), dtype=np.int32)
    table[tuple(xyz.T)] = np.arange(len(xyz), dtype=np.int32)
    padded = np.pad(table, 1, constant_values=len(xyz))
    return np.column_stack([padded[tuple((xyz + offset + 1).T)] for offset in OFFSETS])


def topology(coords):
    """Produce fine-to-coarse links, ordered neighbors, and the stride-2 pooling map."""
    if coords.ndim != 2 or coords.shape[1] != 4 or not len(coords) or coords.dtype.kind not in "iu":
        raise ValueError("Expected integer coordinates [N,4] with columns batch, D, H, W")
    if np.any(coords[:, 0] != 0) or np.any((coords[:, 1:] < 0) | (coords[:, 1:] >= 64)):
        raise ValueError("SLAT accepts batch zero and coordinates in a 64³ grid")
    xyz = coords[:, 1:].astype(np.int32)
    if len(np.unique(xyz, axis=0)) != len(xyz):
        raise ValueError("SLAT coordinates must be unique")
    coarse, parent, counts = np.unique(xyz // 2, axis=0, return_inverse=True, return_counts=True)
    children = np.full((len(coarse), 8), len(xyz), dtype=np.int32)
    phase = (xyz % 2) @ np.array([4, 2, 1], dtype=np.int32)
    children[parent, phase] = np.arange(len(xyz), dtype=np.int32)
    fine = np.column_stack((parent, neighbors(xyz, 64)))
    # The upstream scatter_reduce uses include_self=True and a zero accumulator.
    groups = np.column_stack((coarse, neighbors(coarse, 32), children, counts + 1))
    return fine, groups


def finite_fp32(value, shape, label):
    if value.shape != shape or value.dtype.kind != "f":
        raise ValueError(f"{label} must be floating point with shape {shape}")
    packed = np.asarray(value, dtype="<f4")
    if not np.isfinite(packed).all():
        raise ValueError(f"{label} must be finite in FP32")
    return packed


def schedule(steps, rescale):
    times = 1 - np.arange(steps + 1, dtype=np.float64) / steps
    return rescale * times / (1 + (rescale - 1) * times)


def sample(probe, backend, profile, stage, weights, noise, condition, negative, work, *,
           coords=None, steps=25, guidance=5.0, rescale=3.0, expected_device=None):
    width, grid = PROFILES[profile]
    if condition.ndim != 2 or not 1 <= len(condition) <= 8192:
        raise ValueError("Expected at least one attention context token")
    condition = finite_fp32(condition, (len(condition), width), "Condition")
    negative = finite_fp32(negative, condition.shape, "Negative condition")
    if stage == "structure":
        dense = finite_fp32(noise, (1, 8, grid, grid, grid), "Structure noise")
        state = dense[0].reshape(8, -1).T
        fine = coarse = np.zeros((1, 1), dtype=np.float32)
    else:
        if coords is None:
            raise ValueError("SLAT requires occupied coordinates")
        fine, coarse = topology(coords)
        state = finite_fp32(noise, (len(coords), 8), "SLAT noise")
    paths = [work / f"{stage}-{name}.f32" for name in ("noise", "condition", "negative", "fine", "coarse", "output")]
    for path, value in zip(paths, (state, condition, negative, fine, coarse)):
        np.asarray(value, dtype="<f4").tofile(path)
    result = subprocess.run([probe, "sample", profile, stage, backend, weights, str(len(state)), str(len(condition)),
                             str(len(coarse)), str(steps), str(guidance), str(rescale), *paths], stdout=sys.stderr,
                            stderr=subprocess.PIPE, text=True)
    sys.stderr.write(result.stderr)
    result.check_returncode()
    guided = np.count_nonzero(schedule(steps, rescale)[:-1] >= 0.5) if np.float32(guidance) else 0
    verify_placement(result.stderr, backend, steps + guided, expected_device)
    output = np.fromfile(paths[-1], dtype="<f4")
    if not np.isfinite(output).all():
        raise ValueError(f"Non-finite {stage} output")
    if stage == "structure":
        return np.ascontiguousarray(output.reshape(-1, 8).T.reshape(noise.shape))
    return output.reshape(state.shape)


class SLATSample(NamedTuple):
    structure: np.ndarray
    logits: np.ndarray
    coordinates: np.ndarray
    sampled_features: np.ndarray
    features: np.ndarray


def generate_slat(flow_probe, structure_probe, profile, backend, context, work, *, structure_weights, decoder_weights,
                  slat_weights, rng, steps=25, expected_device=None):
    _, grid = PROFILES[profile]
    zeros = np.zeros_like(context)
    dense = sample(flow_probe, backend, profile, "structure", structure_weights,
                   rng.standard_normal((1, 8, grid, grid, grid), dtype=np.float32), context, zeros, work,
                   steps=steps, expected_device=expected_device)
    logits = decode_structure(structure_probe, "official" if profile == "official" else "fixture", backend,
                              decoder_weights, dense, work, "occupancy", expected_device=expected_device)
    coords = coordinates(logits)
    if not len(coords):
        raise ValueError("The structure decoder returned no occupied voxels")
    latent = sample(flow_probe, backend, profile, "slat", slat_weights,
                    rng.standard_normal((len(coords), 8), dtype=np.float32), context, zeros, work,
                    coords=coords, steps=steps, expected_device=expected_device)
    return SLATSample(dense, logits, coords, latent, latent * SLAT_STD + SLAT_MEAN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="operation", required=True)
    convert = actions.add_parser("convert")
    convert.add_argument("--probe", type=Path, required=True)
    convert.add_argument("--stage", choices=tuple(SCHEMAS), required=True)
    convert.add_argument("--checkpoint", type=Path, required=True)
    convert.add_argument("--weights", type=Path, required=True)
    generate = actions.add_parser("generate")
    for option in ("probe", "structure-probe", "structure-flow", "structure-decoder", "slat-flow", "condition", "output"):
        generate.add_argument("--" + option, type=Path, required=True)
    generate.add_argument("--backend", choices=PLATFORMS, default="cpu")
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--steps", type=int, default=25)
    args = parser.parse_args()
    args.probe = args.probe.resolve(strict=True)
    if args.operation == "convert":
        if args.weights.exists() or args.weights.suffix.lower() != ".safetensors":
            parser.error("--weights must be a new .safetensors file")
        weights = read_checkpoint(args.checkpoint, schema_for(args.probe, "official", args.stage))
        save_file(weights, args.weights, metadata={"architecture": "TRELLIS-image-large", "stage": args.stage, "storage": "F32"})
        return
    if args.output.exists() or args.output.suffix.lower() != ".npz":
        parser.error("--output must be a new .npz file")
    context = np.load(args.condition, allow_pickle=False)
    if not isinstance(context, np.ndarray):
        parser.error("--condition must be a .npy array")
    with scratch_directory() as work:
        os.environ.update(run_environment(work, FORK_TOOLS))
        latent = generate_slat(args.probe, args.structure_probe.resolve(strict=True), "official", args.backend, context, work,
                               structure_weights=args.structure_flow, decoder_weights=args.structure_decoder,
                               slat_weights=args.slat_flow, rng=np.random.default_rng(args.seed), steps=args.steps)
        with args.output.open("xb") as output:
            np.savez(output, coordinates=latent.coordinates, features=latent.features)


if __name__ == "__main__":
    main()
