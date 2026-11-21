"""Convert the local RealPLKSR checkpoint and check its ZML output against PyTorch."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from safetensors import safe_open


MODEL = "4xNomos2_realplksr_dysample"
PJRT = {"cpu": "cpu", "cuda": "iree_cuda", "rocm": "iree_rocm", "vulkan": "iree_vulkan"}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reference(args):
    import spandrel
    import torch
    from safetensors.torch import save_file

    if min(args.width, args.height) < 1:
        raise ValueError("Image dimensions must be positive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    key = "params_ema" if "params_ema" in checkpoint else "params"
    state = checkpoint[key]
    descriptor = spandrel.ModelLoader().load_from_state_dict(state)
    model = descriptor.model.eval().float()
    if descriptor.architecture.id != "PLKSR" or type(model).__name__ != "RealPLKSR" or descriptor.scale != 4:
        raise ValueError("Expected RealPLKSR x4")
    if descriptor.tags != ["64dim", "28nb", "17ks", "Real", "DySample"] or len(state) != 346:
        raise ValueError(f"Unsupported checkpoint profile: {descriptor.tags}, {len(state)} tensors")
    if not all(isinstance(value, torch.Tensor) and value.dtype == torch.float32 for value in state.values()):
        raise ValueError("Expected FP32 checkpoint tensors")

    y, x = np.indices((args.height, args.width))
    rgb = np.stack(((29 * x + 7 * y) % 256, (13 * y + 32 * (x % 2)) % 256, (x * y * 19) % 256))
    inputs = np.ascontiguousarray(rgb[None], dtype=np.float32) / np.float32(255)
    with torch.inference_mode():
        expected = model(torch.from_numpy(inputs)).numpy()

    args.output.mkdir(parents=True, exist_ok=False)
    input_path = args.output / "input.f32le"
    inputs.astype("<f4", copy=False).tofile(input_path)
    expected.astype("<f4", copy=False).tofile(args.output / "expected.f32le")
    metadata = {"model": MODEL, "checkpoint_sha256": sha256(args.checkpoint), "checkpoint_key": key,
                "precision": "float32"}
    save_file({name: tensor.detach().contiguous() for name, tensor in state.items()},
              args.output / "model.safetensors", metadata)
    manifest = {
        **metadata, "spandrel_version": spandrel.__version__, "torch_version": str(torch.__version__),
        "width": args.width, "height": args.height, "input_sha256": sha256(input_path),
    }
    (args.output / "reference.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"PyTorch reference: {args.output}")


def compare(args):
    if not np.isfinite(args.tolerance) or args.tolerance < 0:
        raise ValueError("Tolerance must be finite and nonnegative")
    manifest = json.loads((args.reference / "reference.json").read_text(encoding="utf-8"))
    weights = args.reference / "model.safetensors"
    with safe_open(weights, framework="numpy") as checkpoint:
        for field in ("model", "checkpoint_sha256", "checkpoint_key", "precision"):
            if (checkpoint.metadata() or {}).get(field) != manifest[field]:
                raise ValueError(f"Weights and reference differ for {field}")
    input_path = args.reference / "input.f32le"
    if sha256(input_path) != manifest["input_sha256"]:
        raise ValueError("Reference input changed")
    width, height = manifest["width"], manifest["height"]
    result = subprocess.run(
        [str(args.probe.resolve()), args.backend, str(weights.resolve()), str(width), str(height),
         str(input_path.resolve()), str(args.output.resolve())],
        capture_output=True,
    )
    sys.stdout.buffer.write(result.stdout)
    sys.stderr.buffer.write(result.stderr)
    result.check_returncode()
    if f"backend={args.backend} pjrt={PJRT[args.backend]} shards=".encode("ascii") not in result.stderr:
        raise AssertionError("Unexpected backend or missing PJRT placement report")
    shape = (1, 3, height * 4, width * 4)
    expected = np.fromfile(args.reference / "expected.f32le", dtype="<f4").reshape(shape)
    actual = np.fromfile(args.output, dtype="<f4").reshape(shape)
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError("Expected finite FP32 output")
    delta = actual.astype(np.float64) - expected
    print(f"FP32 max_abs={np.max(np.abs(delta)):.7g} rmse={np.sqrt(np.mean(delta ** 2)):.7g}")
    np.testing.assert_allclose(actual, expected, rtol=args.tolerance, atol=args.tolerance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    generate = commands.add_parser("reference")
    generate.add_argument("--checkpoint", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--width", type=int, default=13)
    generate.add_argument("--height", type=int, default=9)
    generate.set_defaults(run=reference)
    verify = commands.add_parser("compare")
    verify.add_argument("--reference", type=Path, required=True)
    verify.add_argument("--probe", type=Path, required=True)
    verify.add_argument("--backend", choices=PJRT, required=True)
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--tolerance", type=float, default=1e-3)
    verify.set_defaults(run=compare)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
