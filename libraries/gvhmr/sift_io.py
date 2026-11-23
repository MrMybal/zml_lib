"""Stream SimpleVO keyframes to the ZML SIFT and matching probe."""

import subprocess
import sys

import numpy as np

from probe_placement import verify_placement


FEATURES, DESCRIPTOR_WIDTH, MATCH_WIDTH = 4096, 131, 5


def placement_count(frames, height, width, repeats):
    side, octaves = 2 * min(height, width), 0
    while side >= 20 and octaves < 8:
        side //= 2
        octaves += 1
    # Base and packing, six outputs per octave, one match output per pair.
    return repeats * (frames * (2 + 6 * octaves) + frames - 1)


def run(probe, backend, images, work, *, repeats=1, expected_device=None, descriptors=False,
        full_resolution=False, capture_reduced=False):
    frames = len(images)
    if not 2 <= frames <= 514:
        raise ValueError("SIFT expects 2–514 keyframes")
    shape = images[0].shape
    if len(shape) != 3 or shape[2] != 3 or any(image.shape != shape or image.dtype != np.uint8 for image in images):
        raise ValueError("SIFT expects equally sized HWC uint8 images with three channels")
    input_height, input_width = shape[:2]
    if not (1 <= input_height <= 8192 and 1 <= input_width <= 8192):
        raise ValueError("SIFT expects input dimensions of 1–8192 pixels")
    height, width = (input_height // 2, input_width // 2) if full_resolution else (input_height, input_width)
    if not (10 <= height <= 4096 and 10 <= width <= 4096 and height * width <= 2_097_152):
        raise ValueError("SIFT expects working dimensions of 10–4096 pixels, up to 2,097,152 pixels per frame")
    if capture_reduced and not full_resolution:
        raise ValueError("A reduced RGB output requires full-resolution keyframes")
    source, target, features = (work / name for name in ("sift-keyframes.u8", "sift-matches.f32", "sift-descriptors.f32"))
    with source.open("wb") as stream:
        for image in images:
            image.tofile(stream)
    command = [probe, *(["half"] if full_resolution else []), backend, str(input_height), str(input_width),
               str(frames), source, target, str(repeats)]
    if capture_reduced:
        reduced = work / "sift-keyframes-reduced.u8"
        command.extend(("--reduced-rgb", reduced))
    if descriptors:
        command.append(features)
    result = subprocess.run(command, stdout=sys.stderr, stderr=subprocess.PIPE, text=True)
    sys.stderr.write(result.stderr)
    result.check_returncode()
    dispatches = placement_count(frames, height, width, repeats)
    resize_outputs = frames * repeats if full_resolution else 0
    verify_placement(result.stderr, backend, dispatches + resize_outputs, expected_device)
    matches = np.fromfile(target, dtype="<f4").reshape(frames - 1, FEATURES, MATCH_WIDTH)
    if not np.isfinite(matches).all() or not np.isin(matches[:, :, 4], (0, 1)).all():
        raise ValueError("SIFT returned invalid correspondences")
    pairs = []
    for rows in matches:
        valid = rows[rows[:, 4] == 1]
        left, right = valid[:, :2], valid[:, 2:4]
        for coordinates in (left, right):
            if (coordinates < 0).any() or (coordinates >= (width, height)).any():
                raise ValueError("SIFT returned an out-of-bounds correspondence")
        pairs.append((left, right))
    parsed = {"pairs": pairs, "dispatches": dispatches, "resize_outputs": resize_outputs, "image_hw": (height, width)}
    if capture_reduced:
        parsed["images"] = np.fromfile(reduced, dtype=np.uint8).reshape(frames, height, width, 3)
    if descriptors:
        parsed["descriptors"] = np.fromfile(features, dtype="<f4").reshape(frames, FEATURES, DESCRIPTOR_WIDTH)
        parsed["raw_matches"] = matches
    return parsed
