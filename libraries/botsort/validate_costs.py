"""Validate the persistent ZML BoT-SORT cost worker and tracker integration."""

import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build import ROOT, run_environment, scratch_directory
from probe_placement import PLATFORMS
from botsort_io import MAX_DETECTIONS, ULTRALYTICS_VERSION, cost_worker, with_botsort_costs, with_costs


def reference(tracks, detections, fuse):
    start = np.maximum(tracks[:, None, :2], detections[None, :, :2])
    end = np.minimum(tracks[:, None, 2:4], detections[None, :, 2:4])
    intersection = np.maximum(end - start, 0).prod(axis=-1)
    track_area = (tracks[:, 2:4] - tracks[:, :2]).prod(axis=-1)
    detect_area = (detections[:, 2:4] - detections[:, :2]).prod(axis=-1)
    union = detect_area[None, :] + track_area[:, None] - intersection
    cost = 1 - intersection / (union + np.float32(1e-7))
    return 1 - (1 - cost) * detections[None, :, 4] if fuse else cost


def fixture(rng, rows, columns):
    starts = rng.uniform(-50, 600, (max(rows, columns), 2)).astype(np.float32)
    size = rng.uniform(8, 120, (len(starts), 2)).astype(np.float32)
    tracks = np.concatenate((starts[:rows], (starts + size)[:rows]), axis=1)
    shifted = starts[:columns] + rng.uniform(-15, 15, (columns, 2)).astype(np.float32)
    scores = rng.uniform(0.1, 0.99, (columns, 1)).astype(np.float32)
    detections = np.concatenate((shifted, shifted + size[:columns], scores), axis=1)
    tracks[0, 2:4] = tracks[0, :2]  # A zero-area predicted box.
    detections[0, :4] = tracks[0]
    return tracks, detections


def validate_session(args, work):
    rng = np.random.default_rng(7201)
    shapes = ((2, 3), (11, 7), (9, 19), (37, 65), (4, 2), (24, 10))
    assignments = 0
    worst = 0.0

    class ReferenceTracker:
        def __init__(self, fuse):
            self.args = SimpleNamespace(with_reid=False, fuse_score=fuse)

        def get_dists(self, tracks, detections):
            t = np.asarray([track.xyxy for track in tracks], dtype=np.float32)
            d = np.asarray([(*item.xyxy, item.score) for item in detections], dtype=np.float32)
            return reference(t, d, self.args.fuse_score)

    with cost_worker(args.probe, args.backend, work, expected_device=args.expected_device) as session:
        track_type = with_costs(ReferenceTracker, session, verify=True)
        plain, fused = track_type(False), track_type(True)
        empty = session.distances(np.empty((0, 4)), np.zeros((2, 5)), fuse=False)
        assert empty.shape == (0, 2) and session.dispatches == 0
        for frame in range(195):
            boxes, observations = fixture(rng, *shapes[frame % len(shapes)])
            use_score = bool(frame % 2)
            tracker = fused if use_score else plain
            tracks = [SimpleNamespace(xyxy=row, angle=None) for row in boxes]
            detections = [SimpleNamespace(xyxy=row[:4], score=row[4], angle=None) for row in observations]
            actual = tracker.get_dists(tracks, detections)
            expected = reference(boxes, observations, use_score)
            worst = max(worst, float(np.max(np.abs(actual - expected))))
            row, col = linear_sum_assignment(actual)
            expected_row, expected_col = linear_sum_assignment(expected)
            np.testing.assert_array_equal(row, expected_row)
            np.testing.assert_array_equal(col, expected_col)
            assignments += int(np.count_nonzero(actual[row, col] <= np.float32(0.8)))
        assert plain.get_dists([], detections).shape == (0, len(detections))
        assert fused.get_dists(tracks, []).shape == (len(tracks), 0)
    expected_capacities = [(8, 8), (16, 8), (16, 32), (64, 128)]
    if session.dispatches != 195 or session.capacities != expected_capacities:
        raise AssertionError(f"Unexpected session reuse: {session.dispatches}, {session.capacities}")
    return {"frames": 195, "native_outputs": session.dispatches, "compile_capacities": session.capacities,
            "matches_at_cost_limit": assignments, "max_abs_error": worst, "empty_matrices_identical": True}


def validate_request_error(args):
    command = [args.probe, args.backend]
    request = struct.pack("<IIi", 1, 1, 0) + np.array([0, 0, np.nan, 2, 0, 0, 2, 2, 0.8], dtype="<f4").tobytes()
    result = subprocess.run(command, input=request, capture_output=True)
    if not result.returncode or result.stdout or b"NonFiniteAssociationInput" not in result.stderr:
        raise AssertionError(f"Nonfinite worker request was not rejected: {result.stderr.decode(errors='replace')}")


def validate_ultralytics(args, work):
    config_dir = work / "ultralytics"
    config_dir.mkdir()
    os.environ.update(YOLO_CONFIG_DIR=str(config_dir), MPLCONFIGDIR=str(work / "matplotlib"),
                      YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false")
    import cv2
    from ultralytics import __version__
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.bot_sort import BOTSORT
    from ultralytics.utils import IterableSimpleNamespace, YAML
    from ultralytics.utils.checks import check_yaml

    if __version__ != ULTRALYTICS_VERSION:
        raise RuntimeError(f"Expected Ultralytics {ULTRALYTICS_VERSION}, got {__version__}")
    config = YAML.load(check_yaml("botsort.yaml"))
    rng = np.random.default_rng(103)
    texture = rng.integers(0, 256, size=(192, 256, 3), dtype=np.uint8)

    def replay(tracker_type):
        cv2.setRNGSeed(103)
        tracker = tracker_type(args=IterableSimpleNamespace(**config), frame_rate=30)
        outputs = []
        for frame in range(195):
            shift = frame / 8
            camera = 2 * (frame // 8)
            boxes = [[40 + shift, 50, 80 + shift, 150, 0.95, 0],
                     [10, 15, 35, 70, 0.18 if frame % 20 < 4 else 0.87, 0]]
            if 15 <= frame < 110:
                boxes.append([62, 24 + shift, 95, 85 + shift, 0.92, 0])
            entries = np.asarray(boxes, dtype=np.float32)
            entries[:, [0, 2]] += camera
            detections = Boxes(entries, texture.shape[:2])
            outputs.append(tracker.update(detections, np.roll(texture, camera, axis=1)))
        assert tracker.gmc.initializedFirstFrame
        return outputs

    expected = replay(BOTSORT)
    with cost_worker(args.probe, args.backend, work, expected_device=args.expected_device) as session:
        tracker_type = with_botsort_costs(session, verify=True)
        actual = replay(tracker_type)
        replay_dispatches = dict(session.stage_dispatches)

        # One actual duplicate and enough lost tracks to force two column chunks.
        active = [SimpleNamespace(xyxy=np.array([0, 0, 10, 10], dtype=np.float32),
                                  angle=None, frame_id=30, start_frame=1)]
        lost = [SimpleNamespace(xyxy=np.array([20 * col, 0, 20 * col + 10, 10], dtype=np.float32),
                                angle=None, frame_id=30, start_frame=15)
                for col in range(MAX_DETECTIONS + 1)]
        expected_active, expected_lost = BOTSORT.remove_duplicate_stracks(active, lost)
        actual_active, actual_lost = tracker_type.remove_duplicate_stracks(active, lost)
        assert actual_active[0] is expected_active[0] is active[0]
        assert [id(track) for track in actual_lost] == [id(track) for track in expected_lost]
        assert len(actual_lost) == MAX_DETECTIONS and actual_lost[0] is lost[1]
        duplicate_chunk_dispatches = session.stage_dispatches["duplicates"] - replay_dispatches.get("duplicates", 0)
        assert duplicate_chunk_dispatches == 2
    for actual_frame, expected_frame in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_frame, expected_frame)
    expected_stages = {"main_and_unconfirmed", "low_confidence", "duplicates"}
    if replay_dispatches.keys() != expected_stages or not all(replay_dispatches.values()):
        raise AssertionError(f"Expected each BoT-SORT stage to dispatch, got {replay_dispatches}")
    return {"version": __version__, "frames": len(actual), "native_outputs": sum(replay_dispatches.values()),
            "stage_dispatches": replay_dispatches, "total_track_rows": sum(len(frame) for frame in actual),
            "tracks_equal": True, "duplicate_chunk_dispatches": duplicate_chunk_dispatches,
            "dedup_large_track_groups_equal": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--backend", choices=PLATFORMS, default="cuda")
    parser.add_argument("--expected-device")
    parser.add_argument("--ultralytics", action="store_true", help="also replay the pinned Ultralytics tracker")
    args = parser.parse_args()
    args.probe = args.probe.resolve(strict=True)
    tools = ROOT.parent / "zml_win" / "bazel-win-tools"
    with redirect_stdout(sys.stderr), scratch_directory() as work:
        os.environ.update(run_environment(work, tools), CUDA_CACHE_PATH=str(work / "cuda-cache"))
        (work / "cuda-cache").mkdir()
        report = validate_session(args, work)
    with redirect_stdout(sys.stderr), scratch_directory() as work:
        os.environ.update(run_environment(work, tools), CUDA_CACHE_PATH=str(work / "cuda-cache"))
        (work / "cuda-cache").mkdir()
        validate_request_error(args)
    report["nonfinite_rejected"] = True
    if args.ultralytics:
        with redirect_stdout(sys.stderr), scratch_directory() as work:
            os.environ.update(run_environment(work, tools), CUDA_CACHE_PATH=str(work / "cuda-cache"))
            (work / "cuda-cache").mkdir()
            report["ultralytics"] = validate_ultralytics(args, work)
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
