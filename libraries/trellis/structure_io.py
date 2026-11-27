"""Convert a local TRELLIS structure decoder or extract occupied voxel coordinates."""

import argparse
from hashlib import sha256
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from safetensors.numpy import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build import ROOT, run_environment, scratch_directory
from probe_placement import PLATFORMS, verify_placement


SCHEMA_SHA256 = "507a646cb52a6cd5f08c98e96f9312a8c0ba20c239dae32158091ad1b3c0e299"
FORK_TOOLS = ROOT.parent / "zml_win" / "bazel-win-tools"


def schema_for(probe, profile):
    lines = subprocess.check_output([probe, "schema", profile], text=True).splitlines()
    schema = {}
    for line in lines:
        name, *dims = line.split("\t")
        if name in schema:
            raise ValueError(f"Duplicate decoder tensor {name}")
        schema[name] = tuple(map(int, dims))
    if profile == "official":
        manifest = "\n".join(name + ":" + ",".join(map(str, shape)) for name, shape in sorted(schema.items()))
        if sha256(manifest.encode("ascii")).hexdigest() != SCHEMA_SHA256:
            raise ValueError("Decoder schema differs from the official checkpoint header")
    return schema


def read_checkpoint(path, schema):
    weights = load_file(path)
    if weights.keys() != schema.keys():
        raise ValueError(f"Decoder checkpoint has unexpected or missing keys: {sorted(weights.keys() ^ schema.keys())}")
    for name, shape in schema.items():
        tensor = weights[name]
        if tensor.shape != shape or tensor.dtype not in (np.float16, np.float32):
            raise ValueError(f"Expected {name} with shape {shape} in F16 or F32")
    return {name: np.ascontiguousarray(tensor, dtype=np.float32) for name, tensor in weights.items()}


def run(probe, profile, backend, weights, latent, work, stem, repeats=1, *, expected_device=None, expected_pjrt=None):
    src, dst = work / f"{stem}.input.f32", work / f"{stem}.output.f32"
    if latent.ndim != 5 or latent.shape[:2] != (1, 8) or any(not 1 <= dim <= 16 for dim in latent.shape[2:]):
        raise ValueError("Expected latent shape [1,8,D,H,W] with edges 1–16")
    if latent.dtype.kind != "f":
        raise ValueError("Expected a finite floating-point latent")
    packed = np.asarray(latent, dtype="<f4")
    if not np.isfinite(packed).all():
        raise ValueError("Expected a finite FP32 latent")
    packed.tofile(src)
    dims = latent.shape[2:]
    result = subprocess.run([probe, "decode", profile, backend, weights, *map(str, dims), src, dst, str(repeats)],
                            stdout=sys.stderr, stderr=subprocess.PIPE, text=True)
    sys.stderr.write(result.stderr)
    result.check_returncode()
    verify_placement(result.stderr, backend, repeats, expected_device, expected_pjrt=expected_pjrt)
    return np.fromfile(dst, dtype="<f4").reshape(1, 1, *(4 * dim for dim in dims))


def coordinates(logits):
    return np.argwhere(logits[:, 0] > 0).astype(np.int32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="operation", required=True)
    convert = actions.add_parser("convert")
    decode = actions.add_parser("decode")
    for action in (convert, decode):
        action.add_argument("--probe", type=Path, required=True)
        action.add_argument("--weights", type=Path, required=True)
    convert.add_argument("--checkpoint", type=Path, required=True)
    decode.add_argument("--latent", type=Path, required=True)
    decode.add_argument("--output", type=Path, required=True)
    decode.add_argument("--backend", choices=PLATFORMS, default="cpu")
    args = parser.parse_args()
    args.probe = args.probe.resolve(strict=True)
    if args.operation == "convert":
        if args.weights.exists() or args.weights.suffix.lower() != ".safetensors":
            parser.error("--weights must be a new .safetensors file")
        weights = read_checkpoint(args.checkpoint, schema_for(args.probe, "official"))
        save_file(weights, args.weights, metadata={"architecture": "TRELLIS SparseStructureDecoder", "storage": "F32"})
        return
    if args.output.exists() or args.output.suffix.lower() != ".npz":
        parser.error("--output must be a new .npz file")
    latent = np.load(args.latent, allow_pickle=False)
    if not isinstance(latent, np.ndarray):
        parser.error("--latent must be a .npy array")
    with scratch_directory() as work:
        os.environ.update(run_environment(work, FORK_TOOLS))
        logits = run(args.probe, "official", args.backend, args.weights, latent, work, "structure")
        with args.output.open("xb") as output:
            np.savez(output, logits=logits, coordinates=coordinates(logits))


if __name__ == "__main__":
    main()
