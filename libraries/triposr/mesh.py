"""Convert local TripoSR weights or reconstruct a colored PLY from an image."""

import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import is_zipfile

import numpy as np
from PIL import Image, ImageOps
from skimage.measure import marching_cubes

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build import ROOT, run_environment, scratch_directory


RADIUS = 0.87
FORK_TOOLS = ROOT.parent / "zml_win" / "bazel-win-tools"
IMAGE_SIZE = {"official": 512, "fixture": 24}
PLANE_SHAPE = {"official": (3, 40, 64, 64), "fixture": (3, 4, 6, 6)}
POOLER = {
    "image_tokenizer.model.pooler.dense.weight": (768, 768),
    "image_tokenizer.model.pooler.dense.bias": (768,),
}


def schema_for(probe, profile):
    lines = subprocess.check_output([probe, "schema", profile], text=True).splitlines()
    schema = {"encoder": {}, "field": {}}
    for line in lines:
        part, name, *shape = line.split("\t")
        schema[part][name] = tuple(map(int, shape))
    if not all(schema.values()):
        raise ValueError("The probe did not report both TripoSR parameter sets")
    return schema


def load_checkpoint(checkpoint, schema):
    import torch

    expected = schema["encoder"] | schema["field"]
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=is_zipfile(checkpoint))
    if not isinstance(weights, dict):
        raise ValueError("Expected the TripoSR state dictionary")
    for name, shape in expected.items():
        tensor = weights.get(name)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape or tensor.dtype != torch.float32:
            raise ValueError(f"Expected FP32 tensor {name} with shape {shape}")
    for name in weights.keys() - expected.keys():
        tensor = weights[name]
        if name not in POOLER or not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != POOLER[name] or tensor.dtype != torch.float32:
            raise ValueError(f"Unexpected checkpoint tensor {name}")
    return weights


def save_weights(weights, schema, directory, profile):
    from safetensors.torch import save_file

    directory.mkdir()
    try:
        for part, entries in schema.items():
            save_file({name: weights[name].contiguous() for name in entries}, directory / f"{part}.safetensors",
                      metadata={"architecture": "TripoSR", "profile": profile})
    except Exception:
        for part in schema:
            (directory / f"{part}.safetensors").unlink(missing_ok=True)
        directory.rmdir()
        raise


def image_tensor(path, size):
    with Image.open(path) as original:
        rgba = np.asarray(ImageOps.exif_transpose(original).convert("RGBA"), dtype=np.float32) / 255
    alpha = rgba[:, :, 3:]
    rgb = rgba[:, :, :3] * alpha + 0.5 * (1 - alpha)
    resized = np.stack([
        np.asarray(Image.fromarray(channel).resize((size, size), Image.Resampling.BILINEAR))
        for channel in rgb.transpose(2, 0, 1)
    ])
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    return (resized - mean) / std


def probe_run(probe, operation, profile, backend, *args):
    subprocess.run([probe, operation, profile, backend, *map(str, args)], check=True, stdout=sys.stderr)


def encode(probe, profile, backend, weights, image, work, repeat=1):
    source, output = work / "image.f32", work / "planes.f32"
    np.asarray(image, dtype="<f4").tofile(source)
    probe_run(probe, "encode", profile, backend, weights / "encoder.safetensors", source, output, repeat)
    return output


def query(probe, profile, backend, weights, planes, points, work):
    source, output = work / "points.f32", work / "field.f32"
    np.asarray(points, dtype="<f4").tofile(source)
    probe_run(probe, "points", profile, backend, weights / "field.safetensors", planes, len(points), source, output)
    return np.fromfile(output, dtype="<f4").reshape(len(points), 4)


def write_ply(path, vertices, faces, colors):
    records = np.empty(len(vertices), dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
    records["xyz"] = vertices
    records["rgb"] = np.rint(np.clip(colors, 0, 1) * 255).astype(np.uint8)
    triangles = np.empty(len(faces), dtype=[("count", "u1"), ("indices", "<i4", 3)])
    triangles["count"] = 3
    triangles["indices"] = faces
    header = (
        f"ply\nformat binary_little_endian 1.0\nelement vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n"
    ).encode("ascii")
    with path.open("xb") as stream:
        stream.write(header)
        records.tofile(stream)
        triangles.tofile(stream)


def extract(probe, profile, backend, weights, planes, output, resolution, level, work):
    if not 2 <= resolution <= 256 or not np.isfinite(level) or level <= 0:
        raise ValueError("Expected resolution 2–256 and a positive finite isosurface level")
    grid_path = work / "density.f32"
    probe_run(probe, "grid", profile, backend, weights / "field.safetensors", planes, resolution, grid_path)
    volume = np.fromfile(grid_path, dtype="<f4").reshape((resolution,) * 3)
    if not np.isfinite(volume).all() or not volume.min() < level < volume.max():
        raise ValueError("Density grid does not contain a finite surface at the chosen level")
    step = 2 * RADIUS / (resolution - 1)
    vertices, faces, _, _ = marching_cubes(volume, level, spacing=(step,) * 3,
                                            gradient_direction="ascent", allow_degenerate=False)
    vertices -= RADIUS
    colors = query(probe, profile, backend, weights, planes, vertices, work)[:, 1:]
    if not np.isfinite(colors).all():
        raise ValueError("Non-finite vertex colors")
    write_ply(output, vertices, faces, colors)
    return len(vertices), len(faces)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    convert = sub.add_parser("convert")
    convert.add_argument("--probe", type=Path, required=True)
    convert.add_argument("--checkpoint", type=Path, required=True)
    convert.add_argument("--weights", type=Path, required=True)
    mesh = sub.add_parser("mesh")
    mesh.add_argument("--probe", type=Path, required=True)
    mesh.add_argument("--backend", choices=("cpu", "cuda", "rocm", "vulkan"), required=True)
    mesh.add_argument("--weights", type=Path, required=True)
    mesh.add_argument("--input", type=Path, required=True)
    mesh.add_argument("--output", type=Path, required=True)
    mesh.add_argument("--resolution", type=int, default=128)
    mesh.add_argument("--threshold", type=float, default=25.0)
    args = parser.parse_args()
    args.probe = args.probe.resolve(strict=True)
    if args.operation == "convert":
        schema = schema_for(args.probe, "official")
        save_weights(load_checkpoint(args.checkpoint, schema), schema, args.weights, "official")
        return
    if args.output.exists() or args.output.suffix.lower() != ".ply":
        parser.error("--output must be a new .ply file")
    with redirect_stdout(sys.stderr), scratch_directory() as work:
        os.environ.update(run_environment(work, FORK_TOOLS))
        planes = encode(args.probe, "official", args.backend, args.weights,
                        image_tensor(args.input, IMAGE_SIZE["official"]), work)
        vertices, faces = extract(args.probe, "official", args.backend, args.weights, planes,
                                  args.output, args.resolution, args.threshold, work)
    print(json.dumps({"path": str(args.output), "vertices": vertices, "faces": faces}))


if __name__ == "__main__":
    main()
