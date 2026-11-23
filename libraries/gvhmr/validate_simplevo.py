"""Validate the two-view solver against known poses and the CPU trajectory."""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from video import scratch_directory, use_local_runtime
import simplevo_io


def scene(case):
    keys = np.array([0, 8, 16, 19], dtype=np.int32)
    rng = np.random.default_rng(430)
    xyz = rng.uniform((-1.9, -1.1, 3.5), (1.9, 1.1, 8.0), size=(192, 3))
    rotations = Rotation.from_euler("xyz", np.arange(len(keys))[:, None] * (0.35, 3.0, -0.4), degrees=True)
    shift = (0, 0, 0) if case == "panoramic" else (0.48, -0.08, 0.12)
    translations = np.arange(len(keys))[:, None] * shift
    width, height, focal = 640, 360, 420
    images = []
    for rotation, translation in zip(rotations, translations):
        rays = rotation.apply(xyz) + translation
        images.append(focal * rays[:, :2] / rays[:, 2:] + (width / 2, height / 2))
    pairs = []
    for a, b in zip(images[:-1], images[1:]):
        b = b.copy()
        outliers = rng.choice(len(a), 30, replace=False)
        b[outliers] = rng.uniform((20, 20), (width - 20, height - 20), size=(len(outliers), 2))
        pairs.append((a, b))
    return keys, pairs, (width, height, focal), rotations, translations


def validate(args, work):
    keys, pairs, camera, rotations, translations = scene(args.case)
    result = simplevo_io.run(args.probe, args.backend, keys, pairs, camera, work,
                             repeats=2, expected_device=args.expected_device, tolerance=0.75)
    reference = Slerp(keys, rotations)(np.arange(keys[-1] + 1)).as_matrix()
    error = simplevo_io.orientation_error(result["rotation"], reference)
    relative = result["relative"]
    kind = 1 if args.case == "panoramic" else 2
    if error["max_deg"] > 0.15 or not np.all(relative[:, 13] == kind) or relative[:, 12].min() < 160:
        raise ValueError(f"{args.case} failed: {error}, (inliers, model)={relative[:, 12:14].tolist()}")
    delta_rotation = rotations[1:] * rotations[:-1].inv()
    delta_translation = translations[1:] - delta_rotation.apply(translations[:-1])
    if kind == 1:
        translation_error = np.linalg.norm(relative[:, 9:12], axis=1).max()
        if translation_error > 1e-4:
            raise ValueError(f"Panoramic test returned translation of magnitude {translation_error}")
        translation = {"max_translation_norm": float(translation_error)}
    else:
        directions = delta_translation / np.linalg.norm(delta_translation, axis=1, keepdims=True)
        recovered = relative[:, 9:12]
        agreement = np.sum(recovered * directions, axis=1) / np.linalg.norm(recovered, axis=1)
        if not np.all(agreement >= 0.995):
            raise ValueError(f"Incorrect relative translation direction: {agreement}")
        translation = {"min_translation_direction_cosine": float(agreement.min())}
    return {"case": args.case, "pose_error_deg": error, "min_inliers": int(relative[:, 12].min()),
            "same_pose_max_abs_error": simplevo_io.compare_trajectory(keys, result), **translation,
            "odometry_dispatches": result["dispatches"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--case", required=True, choices=("panoramic", "essential"))
    parser.add_argument("--backend", required=True, choices=("cpu", "cuda", "rocm", "vulkan"))
    parser.add_argument("--expected-device")
    args = parser.parse_args()
    args.probe = args.probe.resolve(strict=True)
    with redirect_stdout(sys.stderr), scratch_directory() as work:
        use_local_runtime(work)
        summary = validate(args, work)
    print(json.dumps(summary, allow_nan=False))


if __name__ == "__main__":
    main()
