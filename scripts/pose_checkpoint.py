"""Convert local pose checkpoints and drive the shared FP32 probe protocol."""

import argparse
from pathlib import Path
import subprocess
import sys
from zipfile import is_zipfile

import numpy as np
from safetensors.torch import save_file

from probe_placement import verify_placement


OUTPUTS = {"vitpose": (17, 64, 48), "hmr2": (1024,)}
COUNTS = {"vitpose": 401, "hmr2": 494}
JOINTS = (17, 3)


def schema(probe, kind):
    fields = {}
    for line in subprocess.check_output([probe, "schema"], text=True).splitlines():
        name, *dims = line.split("\t")
        if not dims or name in fields:
            raise ValueError(f"Invalid parameter declaration {line!r}")
        fields[name] = tuple(map(int, dims))
    if len(fields) != COUNTS[kind]:
        raise ValueError(f"Unexpected parameter count for {kind}")
    return fields


def convert(probe, kind, checkpoint, target):
    import torch

    expected = schema(probe, kind)
    loaded = torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=is_zipfile(checkpoint))
    if not isinstance(loaded, dict) or not isinstance(loaded.get("state_dict"), dict):
        raise ValueError(f"Expected a {kind} state_dict checkpoint")
    prefixes = ("backbone.", "keypoint_head.") if kind == "vitpose" else ("backbone.", "smpl_head.transformer.")
    values = {key: value for key, value in loaded["state_dict"].items()
              if key.startswith(prefixes) and not key.endswith(".num_batches_tracked")}
    if values.keys() != expected.keys():
        raise ValueError(f"Mismatched {kind} tensors: {sorted(values.keys() ^ expected.keys())}")
    for name, shape in expected.items():
        if not isinstance(values[name], torch.Tensor) or tuple(values[name].shape) != shape or values[name].dtype != torch.float32:
            raise ValueError(f"Expected FP32 {name} with shape {shape}")
        values[name] = values[name].contiguous()
    save_file(values, target, metadata={"architecture": kind, "storage": "F32"})


def _run_buffer(probe, mode, backend, weights, data, shape, work, label, repeats, batch, expected_device):
    source, output = (work / f"{label}-{name}.f32" for name in ("input", "output"))
    data.astype("<f4", copy=False).tofile(source)
    cmd = [probe, *([mode] if mode else []), backend, *([weights] if weights else []),
           str(len(data)), str(batch), source, output, str(repeats)]
    result = subprocess.run(cmd, stdout=sys.stderr, stderr=subprocess.PIPE, text=True)
    sys.stderr.write(result.stderr)
    result.check_returncode()
    dispatches = repeats * ((len(data) + batch - 1) // batch)
    verify_placement(result.stderr, backend, dispatches, expected_device)
    values = np.fromfile(output, dtype="<f4").reshape(len(data), *shape)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} returned nonfinite values")
    return values, dispatches


def _pack_boxes(data, boxes):
    if boxes.shape != (len(data), 3) or boxes.dtype != np.float32 or not np.isfinite(boxes).all() or (boxes[:, 2] <= 0).any():
        raise ValueError("Expected finite FP32 crop centers and positive crop sizes in [N,3]")
    return np.concatenate((data.reshape(len(data), -1), boxes), axis=1)


def run(probe, kind, backend, weights, images, work, *, repeats=1, batch=4, expected_device=None,
        boxes=None, keep_heatmaps=False):
    if images.ndim != 4 or images.shape[1:] != (3, 256, 192) or not 1 <= len(images) <= 4096:
        raise ValueError("Expected 1–4096 normalized NCHW RGB crops of size 256×192")
    if images.dtype != np.float32 or not np.isfinite(images).all():
        raise ValueError("Expected finite FP32 crops")
    if boxes is None:
        if keep_heatmaps:
            raise ValueError("Keeping heatmaps with decoded joints requires crop boxes")
        return _run_buffer(probe, None, backend, weights, images, OUTPUTS[kind], work, kind, repeats, batch, expected_device)
    if kind != "vitpose":
        raise ValueError("UDP joint decoding requires ViTPose")
    shape = (int(np.prod(JOINTS) + np.prod(OUTPUTS[kind])),) if keep_heatmaps else JOINTS
    mode = "verify-joints" if keep_heatmaps else "joints"
    return _run_buffer(probe, mode, backend, weights, _pack_boxes(images, boxes), shape, work, kind, repeats, batch, expected_device)


def run_udp(probe, backend, heatmaps, boxes, work, *, repeats=1, batch=4, expected_device=None):
    if heatmaps.ndim != 4 or heatmaps.shape[1:] != OUTPUTS["vitpose"] or not 1 <= len(heatmaps) <= 4096:
        raise ValueError("Expected 1–4096 heatmap tensors of shape [17,64,48]")
    if heatmaps.dtype != np.float32 or not np.isfinite(heatmaps).all():
        raise ValueError("Expected finite FP32 heatmaps")
    return _run_buffer(probe, "udp", backend, None, _pack_boxes(heatmaps, boxes), JOINTS, work, "udp", repeats, batch, expected_device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=OUTPUTS, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    if args.weights.exists() or args.weights.suffix != ".safetensors":
        parser.error("--weights must be a new .safetensors file")
    convert(args.probe.resolve(strict=True), args.model, args.checkpoint.resolve(strict=True), args.weights)


if __name__ == "__main__":
    main()
