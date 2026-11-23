"""Convert TRELLIS v1 decoder weights and produce native mesh, Gaussian or radiance outputs."""

import argparse
from hashlib import sha256
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from safetensors.numpy import save_file

from flow_io import FORK_TOOLS, finite_fp32, neighbors
from mesh import extract_mesh
from structure_io import read_checkpoint
from build import run_environment, scratch_directory
from probe_placement import PLATFORMS, verify_placement


SCHEMAS = {
    "mesh": "9d08fd7eb7266db8f34abaa5309e6606cbb5088de50410b8ed91f7205828751a",
    "gaussian": "e8dd8e03e9575f3627be7eec0836b15ed67b44f222d6407d33dca0186f70313b",
    "radiance": "dabb08e5c3755c9ed2e21dd10ef4d9621cc04b59bf6d0ac112faa1afcacbbfb8",
}
PROFILES = {"official": 64, "fixture": 8}
PLY_FIELDS = "x y z nx ny nz f_dc_0 f_dc_1 f_dc_2 opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3".split()
CHILDREN = np.indices((2, 2, 2), dtype=np.int32).reshape(3, -1).T


def schema_for(probe, profile, kind):
    schema = {}
    for line in subprocess.check_output([probe, "schema", profile, kind], text=True).splitlines():
        name, *dims = line.split("\t")
        if name in schema:
            raise ValueError(f"Duplicate representation tensor {name}")
        schema[name] = tuple(map(int, dims))
    if profile == "official":
        manifest = "\n".join(name + ":" + ",".join(map(str, shape)) for name, shape in sorted(schema.items()))
        if sha256(manifest.encode("ascii")).hexdigest() != SCHEMAS[kind]:
            raise ValueError(f"{kind} schema differs from the official checkpoint")
    return schema


def window_table(xyz, shift):
    _, group = np.unique((xyz + shift) // 8, axis=0, return_inverse=True)
    group = group.reshape(-1)
    counts = np.bincount(group)
    sort = np.argsort(group, kind="stable")
    position = np.arange(len(xyz)) - np.repeat(np.cumsum(counts) - counts, counts)
    table = np.full((len(counts), counts.max()), len(xyz), dtype=np.int32)
    table[group[sort], position] = sort
    restore = np.empty(len(xyz), dtype=np.int32)
    restore[sort] = group[sort] * table.shape[1] + position
    return table, restore


def subdivide(cells):
    return (2 * cells[:, None, :] + CHILDREN).reshape(-1, 3)


def decode(probe, profile, kind, backend, weights, coords, features, work, *, repeats=1, expected_device=None):
    edge = PROFILES[profile]
    if coords.ndim != 2 or coords.shape != (len(coords), 4) or not len(coords) or coords.dtype.kind not in "iu":
        raise ValueError("Expected integer coordinates [N,4] with columns batch, X, Y, Z")
    if np.any(coords[:, 0] != 0) or np.any((coords[:, 1:] < 0) | (coords[:, 1:] >= edge)):
        raise ValueError(f"Expected batch zero and coordinates in a {edge}³ grid")
    coords = coords[:, 1:].astype(np.int32)
    if len(np.unique(coords, axis=0)) != len(coords):
        raise ValueError("Duplicate SLAT coordinates")
    features = finite_fp32(features, (len(coords), 8), "SLAT")
    windows, inverse = zip(*(window_table(coords, shift) for shift in (0, 4)))
    files = [work / f"{kind}-{name}.f32" for name in ("features", "coords", "windows0", "windows1", "inverse0", "inverse1",
                                                      "neighbors0", "neighbors1", "result")]
    for path, data in zip(files, (features, coords, *windows, *inverse)):
        np.asarray(data, dtype="<f4").tofile(path)
    cells = coords
    for i, path in enumerate(files[6:8], start=1):
        if kind == "mesh":
            cells = subdivide(cells)
            np.asarray(neighbors(cells, edge << i), dtype="<f4").tofile(path)
        else:
            np.zeros(1, dtype="<f4").tofile(path)
    sizes = (str(dim) for table in windows for dim in table.shape)
    result = subprocess.run([probe, "decode", profile, kind, backend, weights, str(len(coords)), *sizes, *files, str(repeats)],
                            stdout=sys.stderr, stderr=subprocess.PIPE, text=True)
    sys.stderr.write(result.stderr)
    result.check_returncode()
    verify_placement(result.stderr, backend, repeats, expected_device)
    width = {"mesh": 101, "gaussian": 448, "radiance": 451}[kind]
    values = np.fromfile(files[-1], dtype="<f4").reshape(len(cells), width)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite TRELLIS {kind} output")
    return cells, values


def gaussian_rows(values):
    xyz = values[:, :96].reshape(-1, 3) - 0.5
    dc = values[:, 96:192].reshape(-1, 3)
    scales = values[:, 192:288].reshape(-1, 3)
    rots = values[:, 288:416].reshape(-1, 4).copy()
    opacity = values[:, 416:].reshape(-1, 1) + np.float32(np.log(0.1 / 0.9))
    scales = np.logaddexp(np.float32(0), scales + np.float32(np.log(np.expm1(0.004))))
    scales = np.log(np.hypot(scales, np.float32(0.0009)))
    rots[:, 0] += 1
    # Convert from the model axes to the default TRELLIS PLY axes.
    xyz = xyz[:, [0, 2, 1]] * np.array([1, -1, 1], np.float32)
    a, b, c, d = rots.T
    rotations = np.column_stack((a - b, a + b, c - d, c + d))
    rotations /= np.maximum(np.linalg.norm(rotations, axis=1, keepdims=True), 1e-12)
    return np.column_stack((xyz, np.zeros_like(xyz), dc, opacity, scales, rotations))


def write_output(kind, path, cells, values, resolution):
    with path.open("xb") as output:
        if kind == "mesh":
            np.savez(output, **extract_mesh(cells, values, resolution * 4))
        elif kind == "radiance":
            np.savez(output, position=values[:, :3], depth=np.full((len(values), 1), resolution.bit_length() - 1, dtype=np.uint8),
                     trivec=values[:, 3:387].reshape(-1, 16, 3, 8), density=values[:, 387:403],
                     features_dc=values[:, 403:].reshape(-1, 16, 1, 3),
                     aabb=np.array([-0.5, -0.5, -0.5, 1, 1, 1], np.float32))
        else:
            properties = "\n".join(f"property float {field}" for field in PLY_FIELDS)
            output.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {32 * len(values)}\n{properties}\nend_header\n".encode("ascii"))
            np.asarray(gaussian_rows(values), dtype="<f4").tofile(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="operation", required=True)
    convert = actions.add_parser("convert")
    run = actions.add_parser("decode")
    for action in (convert, run):
        action.add_argument("--format", choices=tuple(SCHEMAS), required=True)
        action.add_argument("--probe", type=Path, required=True)
        action.add_argument("--weights", type=Path, required=True)
    convert.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--slat", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--backend", choices=PLATFORMS, default="cpu")
    args = parser.parse_args()
    probe = args.probe.resolve(strict=True)
    if args.operation == "convert":
        if args.weights.exists() or args.weights.suffix.lower() != ".safetensors":
            parser.error("--weights must be a new .safetensors file")
        weights = read_checkpoint(args.checkpoint, schema_for(probe, "official", args.format))
        save_file(weights, args.weights, metadata={"architecture": "TRELLIS-image-large", "representation": args.format, "storage": "F32"})
        return
    suffix = ".ply" if args.format == "gaussian" else ".npz"
    if args.output.exists() or args.output.suffix.lower() != suffix:
        parser.error(f"--output must be a new {suffix} file")
    with np.load(args.slat, allow_pickle=False) as archive, scratch_directory() as work:
        os.environ.update(run_environment(work, FORK_TOOLS))
        cells, values = decode(probe, "official", args.format, args.backend, args.weights, archive["coordinates"], archive["features"], work)
        write_output(args.format, args.output, cells, values, PROFILES["official"])


if __name__ == "__main__":
    main()
