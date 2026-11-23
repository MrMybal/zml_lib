"""Compare native GVHMR temporal conditions against PyTorch without model weights."""

import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build import ROOT, run_environment, scratch_directory
from probe_placement import PLATFORMS
import conditions_io


def fixture(frames, moving):
    rng = np.random.default_rng(3187)
    boxes = rng.uniform(10, 960, (frames, 3)).astype(np.float32)
    centers = boxes[:, None, :2]
    radius = boxes[:, None, 2:] * np.float32(0.5)
    points = rng.uniform(-0.25, 1.0, (frames, 17, 3)).astype(np.float32)
    points[..., :2] = centers + points[..., :2] * radius * np.float32(2)
    points[:, :2, :2] = np.concatenate((centers - radius, centers + radius), axis=1)
    points[:, 2, :2] = points[:, 1, :2]
    points[:, 2, 0] = np.nextafter(points[:, 2, 0], np.float32(np.inf))
    points[:, 3, :2] = points[:, 0, :2]
    points[:, 3, 1] = np.nextafter(points[:, 3, 1], np.float32(-np.inf))
    points[:, :4, 2] = np.float32(0.75)
    points[:, 4, :2] = centers[:, 0]
    points[:, 4, 2] = np.float32(-0.125)
    camera = np.tile(np.eye(3, dtype=np.float32), (frames, 1, 1))
    camera[:, 0, 0] = rng.uniform(420, 3000, frames)
    camera[:, 1, 1] = rng.uniform(420, 3000, frames)
    camera[:, :2, 2] = rng.uniform(100, 1000, (frames, 2))
    angular = np.tile(np.array([1, 0, 0, 0, 1, 0], dtype=np.float32), (frames, 1))
    stats = None
    if moving:
        angular += rng.normal(0, 0.034, (frames, 6)).astype(np.float32)
        stats = (np.array([1, 0, 0, 0, 1, 0], dtype=np.float32),
                 rng.uniform(0.005, 0.13, 6).astype(np.float32))
    return (points, boxes, camera, angular), stats


def reference(data, stats):
    points, boxes, camera, angular = map(torch.from_numpy, data)
    center = boxes[:, None, :2]
    side = boxes[:, None, 2:]
    xy = points[..., :2]
    inside = ((xy >= center - side / 2) & (xy <= center + side / 2)).all(dim=-1, keepdim=True)
    obs = torch.cat((2 * (xy - center) / side, points[..., 2:] * inside), dim=-1)
    offsets = boxes.clone()
    offsets[:, :2] -= camera[:, :2, 2]
    cliffcam = offsets / camera[:, :1, 0]
    if stats is not None:
        mean, deviation = map(torch.from_numpy, stats)
        angular = (angular - mean) / deviation
    return {"obs": obs.numpy(), "f_cliffcam": cliffcam.numpy(), "f_cam_angvel": angular.numpy()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--backend", choices=PLATFORMS, default="cuda")
    parser.add_argument("--expected-device")
    args = parser.parse_args()
    probe = args.probe.resolve(strict=True)
    reports = {}
    for frames, moving in ((1, False), (195, True), (4096, True)):
        with redirect_stdout(sys.stderr), scratch_directory() as work:
            cache = work / "cuda-cache"
            cache.mkdir()
            os.environ.update(run_environment(work, ROOT.parent / "zml_win" / "bazel-win-tools"), CUDA_CACHE_PATH=str(cache))
            print(f"GVHMR conditions: {frames} synthetic frames", flush=True)
            data, stats = fixture(frames, moving)
            result = conditions_io.run(probe, args.backend, *data, work, stats=stats, repeats=2,
                                       expected_device=args.expected_device)
            errors = conditions_io.compare(result, reference(data, stats))
            boundary_scores = np.tile([0.75, 0.75, 0.0, 0.0, -0.125], (frames, 1))
            np.testing.assert_array_equal(result["obs"][:, :5, 2], boundary_scores)
            reports[frames] = {"max_abs_error": errors, "native_outputs": 2, "confidence_identical": True,
                               "angular_normalized": moving}
    print(json.dumps(reports, allow_nan=False))


if __name__ == "__main__":
    main()
