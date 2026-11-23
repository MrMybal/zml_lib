"""Compare the weight-free ZML UDP decoder with the local GVHMR OpenCV reference."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build import run_environment, scratch_directory
from pose_checkpoint import run_udp
from probe_placement import PLATFORMS


def fixtures():
    y, x = np.mgrid[:64, :48]
    centers = [(22.37, 30.17), (0.04, 22.3), (47.1, 25.26), (18.7, 0.2),
               (21.9, 63.31), (0.1, 0.2), (47.25, 63.37)]
    maps = np.empty((len(centers), 17, 64, 48), dtype=np.float32)
    for i, (cx, cy) in enumerate(centers):
        for j in range(maps.shape[1]):
            dx, dy = x - cx - 0.025 * j, y - cy + 0.018 * j
            maps[i, j] = np.exp(-(dx * dx + 1.3 * dy * dy + 0.26 * dx * dy) / 15) * (0.6 + 0.02 * j)
    # Equal maxima must select the first position in row-major order.
    maps[0, 0] = 0
    maps[0, 0, 20, 8] = maps[0, 0, 35, 38] = 1
    maps[2, 5] = 80  # Log clamp and zero curvature.
    maps[4, 12] = 0
    maps[-1, -1] = -1  # An absent joint adjacent to a valid heatmap.
    boxes = np.array([[180 + 4 * i, 317 - 2 * i, 172 + 47 * i] for i in range(len(centers))], dtype=np.float32)
    return maps, boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--backend", choices=PLATFORMS, default="cuda")
    parser.add_argument("--expected-device")
    args = parser.parse_args()

    sys.path.insert(0, str(args.upstream.resolve(strict=True)))
    from hmr4d.utils.kpts.kp2d_utils import keypoints_from_heatmaps

    maps, boxes = fixtures()
    sizes = boxes[:, 2:] * np.array([0.75, 1], dtype=np.float32)
    expected, scores = keypoints_from_heatmaps(maps, center=boxes[:, :2], scale=sizes / 200, use_udp=True)
    visible = scores[..., 0] > 0
    # The port keeps the (-1, -1) heatmap sentinel for nonpositive scores.
    extent = sizes / 200 * 200
    absent = boxes[:, :2] - extent * 0.5 - extent / np.array([47, 63], dtype=np.float32)
    expected = np.where(visible[..., None], expected, absent[:, None])
    with scratch_directory() as work:
        tools = Path(__file__).resolve().parents[3] / "zml_win" / "bazel-win-tools"
        os.environ.update(run_environment(work, tools))
        actual, dispatches = run_udp(args.probe.resolve(strict=True), args.backend, maps, boxes, work,
                                     repeats=2, expected_device=args.expected_device)
        np.testing.assert_array_equal(actual[..., 2], scores[..., 0])
        np.testing.assert_allclose(actual[..., :2], expected, atol=0.002, rtol=0, equal_nan=False)
        print(json.dumps({"udp_dispatches": dispatches, "joints": int(visible.size),
                          "absent_joints": int(np.count_nonzero(~visible)),
                          "max_abs_error_px": float(np.max(np.abs(actual[..., :2] - expected)))}))


if __name__ == "__main__":
    main()
