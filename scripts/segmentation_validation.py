"""Shared FP32 probe and mask comparison for the two SAM libraries."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np


PLATFORMS = {"cpu": "cpu", "cuda": "iree_cuda", "rocm": "iree_rocm", "vulkan": "iree_vulkan"}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_rgb(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def check_checkpoint(path, expected, source_hash):
    from safetensors import safe_open
    import torch

    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        if checkpoint.metadata()["checkpoint_sha256"] != source_hash:
            raise ValueError(f"Checkpoint provenance differs for {path}")
        if set(checkpoint.keys()) != set(expected):
            raise ValueError(f"Checkpoint tensor names differ for {path}")
        for name, value in expected.items():
            if not torch.equal(checkpoint.get_tensor(name), value):
                raise ValueError(f"Checkpoint differs at {name}")


def probe(args, command, weights, data, shape, label, count=0):
    src = (args.output / f"{label}-input.f32").resolve()
    dst = (args.output / f"{label}-output.f32").resolve()
    np.asarray(data, dtype="<f4").tofile(src)
    process = subprocess.run([args.probe.resolve(), args.backend, command, weights.resolve(), str(count), src, dst],
                             capture_output=True)
    sys.stdout.buffer.write(process.stdout)
    sys.stderr.buffer.write(process.stderr)
    process.check_returncode()
    if f"backend={args.backend} pjrt={PLATFORMS[args.backend]}".encode("ascii") not in process.stderr:
        raise AssertionError("Execution placement does not match the requested backend")
    return np.fromfile(dst, dtype="<f4").reshape(shape)


def compare(name, actual, expected, atol, rtol):
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError(f"{name} contains incompatible or nonfinite values")
    delta = actual.astype(np.float64) - expected
    max_error = float(np.max(np.abs(delta)))
    print(f"{name}: max_abs={max_error:.7g} rmse={np.sqrt(np.mean(delta ** 2)):.7g}", flush=True)
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=name)
    return max_error


def raster_iou(actual, expected):
    intersection = np.sum(actual & expected, axis=(1, 2))
    union = np.sum(actual | expected, axis=(1, 2))
    ious = np.divide(intersection, union, out=np.ones_like(union, dtype=float), where=union > 0)
    if min(ious) < 0.98:
        raise AssertionError(f"Mask IoU below 0.98: {ious}")
    return float(min(ious))


def write_report(args, metrics):
    data = {"backend": args.backend, "checkpoint_sha256": sha256(args.checkpoint),
            "image_sha256": sha256(args.image), **metrics}
    (args.output / "report.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(data), flush=True)


def arguments(parser, video=False):
    actions = parser.add_subparsers(dest="action", required=True)
    conv = actions.add_parser("convert")
    verify = actions.add_parser("compare")
    for p in (conv, verify):
        p.add_argument("--checkpoint", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
    for key in ("weights", "image", "probe", "reference"):
        verify.add_argument(f"--{key}", type=Path, required=True)
    verify.add_argument("--backend", choices=PLATFORMS, default="cpu")
    if video:
        verify.add_argument("--video", action="store_true")
    return parser.parse_args()
