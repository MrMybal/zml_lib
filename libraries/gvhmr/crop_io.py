"""Run the shared ViTPose-H/HMR2 crop graph on decoded video frames."""

import re
import subprocess
import sys

import numpy as np

from probe_placement import verify_placement


def _run(probe, backend, images, boxes, work, *, half, capture_half, repeats, batch, expected_device):
    if not 1 <= len(images) <= 4096:
        raise ValueError("Expected 1–4096 RGB8 frames")
    shape = images[0].shape
    if len(shape) != 3 or shape[-1] != 3 or any(frame.shape != shape or frame.dtype != np.uint8 for frame in images):
        raise ValueError("Expected equally sized RGB8 frames in [H,W,3]")
    if boxes.shape != (len(images), 3) or boxes.dtype != np.float32 or not np.isfinite(boxes).all() or (boxes[:, 2] <= 0).any():
        raise ValueError("Expected finite FP32 crop boxes with positive sizes in [N,3]")
    h, w = shape[:2]
    frames_file, boxes_file, output = (work / f"pose-crop-{name}" for name in ("frames.rgb", "boxes.f32", "output.f32"))
    with frames_file.open("xb") as stream:
        for frame in images:
            frame.tofile(stream)
    boxes.astype("<f4", copy=False).tofile(boxes_file)
    reduced_file = work / "pose-crop-half.rgb" if capture_half else None
    cmd = [probe, *(["half"] if half else []), backend, str(h), str(w), str(len(images)), str(batch),
           frames_file, boxes_file, output, str(repeats), *([reduced_file] if capture_half else [])]
    result = subprocess.run(cmd, stdout=sys.stderr, stderr=subprocess.PIPE, text=True)
    sys.stderr.write(result.stderr)
    result.check_returncode()
    dispatches = repeats * ((len(images) + batch - 1) // batch)
    verify_placement(result.stderr, backend, dispatches * (2 if capture_half else 1), expected_device)
    counts = re.findall(r"\bpose_crop_frames=(\d+) antialiased=(\d+)$", result.stderr, re.MULTILINE)
    if len(counts) != 1 or int(counts[0][0]) != len(images):
        raise RuntimeError("Missing or inconsistent pose crop diagnostics")
    crops = np.fromfile(output, dtype="<f4").reshape(len(images), 3, 256, 192)
    if not np.isfinite(crops).all():
        raise ValueError("The native pose crop contains nonfinite pixels")
    reduced = None
    if reduced_file is not None:
        reduced = np.fromfile(reduced_file, dtype=np.uint8).reshape(len(images), max(1, h // 2), max(1, w // 2), 3)
    return crops, reduced, dispatches, int(counts[0][1])


def run(probe, backend, images, boxes, work, *, repeats=1, batch=4, expected_device=None):
    crops, _, dispatches, filtered = _run(
        probe, backend, images, boxes, work, half=False, capture_half=False,
        repeats=repeats, batch=batch, expected_device=expected_device,
    )
    return crops, dispatches, filtered


def run_half(probe, backend, images, boxes, work, *, capture_half=False, repeats=1, batch=4, expected_device=None):
    return _run(probe, backend, images, boxes, work, half=True, capture_half=capture_half,
                repeats=repeats, batch=batch, expected_device=expected_device)
