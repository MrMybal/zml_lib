"""Convert local DINOv2 ViT-L/14 registers weights or encode an RGB TRELLIS image."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build import ROOT, run_environment, scratch_directory
from probe_placement import PLATFORMS, verify_placement


TOOLS = ROOT.parent / "zml_win" / "bazel-win-tools"
PROFILES = {"registered": (1024, 37, 24, 16), "fixture": (24, 5, 2, 3)}
BACKENDS = tuple(PLATFORMS)


def schema_for(probe, profile):
    schema = {}
    for line in subprocess.check_output([probe, "schema", profile], text=True).splitlines():
        name, *dims = line.split("\t")
        if name in schema or not name.startswith("pretrained."):
            raise ValueError(f"Unexpected DINOv2 name {name}")
        schema[name] = tuple(map(int, dims))
    width, grid, layers, _ = PROFILES[profile]
    if len(schema) != 8 + 14 * layers or schema.get("pretrained.pos_embed") != (1, 1 + grid * grid, width):
        raise ValueError("Unexpected DINOv2 register schema")
    return schema


def read_checkpoint(path, schema):
    import torch
    from safetensors.torch import load_file

    if path.suffix.lower() == ".pth":
        source = torch.load(path, map_location="cpu", weights_only=True)
    elif path.suffix.lower() == ".safetensors":
        source = load_file(path)
    else:
        raise ValueError("Expected a .pth or .safetensors state dict")
    expected = {name.removeprefix("pretrained."): name for name in schema}
    if not isinstance(source, dict) or source.keys() != expected.keys():
        raise ValueError("Checkpoint must match the official ViT-L/14 state dict with four register tokens")
    weights = {}
    for original, name in expected.items():
        tensor = source[original]
        if tensor.shape != schema[name] or tensor.dtype not in (torch.float32, torch.float16):
            raise ValueError(f"Expected {original} with shape {schema[name]} in F16 or F32")
        weights[name] = tensor.float().contiguous().numpy()
    return weights


def encode(probe, profile, backend, weights, image, work, *, repeats=1, expected_device=None):
    if image.ndim != 4 or image.shape[:2] != (1, 3):
        raise ValueError("Expected a single NCHW RGB image")
    h, w = image.shape[2:]
    if any(dim < 14 or dim > 518 or dim % 14 for dim in (h, w)):
        raise ValueError("Image dimensions must be multiples of 14 up to 518")
    if image.dtype.kind != "f" or not np.isfinite(image).all() or np.any((image < 0) | (image > 1)):
        raise ValueError("RGB values must be floating point in [0,1]")
    src, dest = work / "rgb.f32", work / "condition.f32"
    np.asarray(image, dtype="<f4").tofile(src)
    result = subprocess.run([probe, "encode", profile, backend, weights, str(h), str(w), src, dest, str(repeats)],
                            stdout=sys.stderr, stderr=subprocess.PIPE, text=True)
    sys.stderr.write(result.stderr)
    result.check_returncode()
    verify_placement(result.stderr, backend, repeats, expected_device)
    tokens = np.fromfile(dest, dtype="<f4").reshape(5 + (h // 14) * (w // 14), PROFILES[profile][0])
    if not np.isfinite(tokens).all():
        raise ValueError("Non-finite DINOv2 output")
    return tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    convert = commands.add_parser("convert")
    run = commands.add_parser("encode")
    for command in (convert, run):
        command.add_argument("--probe", type=Path, required=True)
        command.add_argument("--weights", type=Path, required=True)
    convert.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--image", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--backend", choices=BACKENDS, default="cpu")
    args = parser.parse_args()
    probe = args.probe.resolve(strict=True)
    if args.operation == "convert":
        if args.weights.exists() or args.weights.suffix.lower() != ".safetensors":
            parser.error("--weights must be a new .safetensors file")
        weights = read_checkpoint(args.checkpoint, schema_for(probe, "registered"))
        save_file(weights, args.weights, metadata={"architecture": "dinov2_vitl14_reg", "storage": "F32"})
        return
    if args.output.exists() or args.output.suffix.lower() != ".npy":
        parser.error("--output must be a new .npy file")
    from PIL import Image

    with Image.open(args.image) as source:
        rgb = np.asarray(source.convert("RGB").resize((518, 518), Image.Resampling.LANCZOS), dtype=np.float32)
    image = rgb.transpose(2, 0, 1)[None] / 255
    with scratch_directory() as work:
        os.environ.update(run_environment(work, TOOLS))
        values = encode(probe, "registered", args.backend, args.weights, image, work)
        with args.output.open("xb") as output:
            np.save(output, values, allow_pickle=False)


if __name__ == "__main__":
    main()
