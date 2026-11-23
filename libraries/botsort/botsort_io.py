"""Stream BoT-SORT association matrices through one native ZML session."""

from collections import Counter
from contextlib import contextmanager
import re
import struct
import subprocess
import sys
from types import FunctionType, SimpleNamespace

import numpy as np

from probe_placement import verify_placement


MAX_TRACKS = 16384
MAX_DETECTIONS = 512
ULTRALYTICS_VERSION = "8.3.253"


class Costs:
    def __init__(self, process, log):
        self.process = process
        self.log = log
        self.dispatches = 0
        self.stage_dispatches = Counter()
        self.capacities = []

    def distances(self, tracks, detections, *, fuse):
        tracks = np.ascontiguousarray(tracks, dtype="<f4")
        detections = np.ascontiguousarray(detections, dtype="<f4")
        if tracks.ndim != 2 or tracks.shape[1] != 4 or detections.ndim != 2 or detections.shape[1] != 5:
            raise ValueError("Expected BoT-SORT tracks [T,4] and detections [D,5]")
        rows, columns = len(tracks), len(detections)
        if rows == 0 or columns == 0:
            return np.empty((rows, columns), dtype=np.float32)
        if rows > MAX_TRACKS or columns > MAX_DETECTIONS or not np.isfinite(tracks).all() or not np.isfinite(detections).all():
            raise ValueError("Invalid BoT-SORT matrix size or coordinates")
        try:
            self.process.stdin.write(struct.pack("<IIi", rows, columns, int(fuse)))
            self.process.stdin.write(memoryview(tracks).cast("B"))
            self.process.stdin.write(memoryview(detections).cast("B"))
            self.process.stdin.flush()
            data = self.process.stdout.read(rows * columns * 4)
        except OSError as error:
            raise RuntimeError(f"BoT-SORT cost worker I/O failed. See {self.log}") from error
        if len(data) != rows * columns * 4:
            raise RuntimeError(f"Incomplete BoT-SORT cost matrix. See {self.log}")
        matrix = np.frombuffer(data, dtype="<f4").reshape(rows, columns).copy()
        if not np.isfinite(matrix).all():
            raise ValueError("The ZML BoT-SORT matrix contains nonfinite values")
        self.dispatches += 1
        return matrix

    def track_distances(self, tracks, detections, *, stage, scores=None):
        rows, columns = len(tracks), len(detections)
        if not rows or not columns:
            return np.empty((rows, columns), dtype=np.float32)
        boxes = np.asarray([track.xyxy for track in tracks], dtype=np.float32)
        observations = np.empty((columns, 5), dtype=np.float32)
        observations[:, :4] = [detection.xyxy for detection in detections]
        observations[:, 4] = 0 if scores is None else scores
        fuse = scores is not None
        before = self.dispatches
        if rows <= MAX_TRACKS and columns <= MAX_DETECTIONS:
            matrix = self.distances(boxes, observations, fuse=fuse)
        else:
            matrix = np.empty((rows, columns), dtype=np.float32)
            for row in range(0, rows, MAX_TRACKS):
                block_rows = slice(row, row + MAX_TRACKS)
                for column in range(0, columns, MAX_DETECTIONS):
                    block_columns = slice(column, column + MAX_DETECTIONS)
                    matrix[block_rows, block_columns] = self.distances(
                        boxes[block_rows], observations[block_columns], fuse=fuse
                    )
        self.stage_dispatches[stage] += self.dispatches - before
        return matrix


@contextmanager
def cost_worker(probe, backend, work, *, expected_device=None):
    log = work / "botsort-cost.log"
    with log.open("xb") as stderr:
        process = subprocess.Popen([probe, backend], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr)
        costs = Costs(process, log)
        try:
            yield costs
            process.stdin.close()
            status = process.wait()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdin.close()
            process.stdout.close()
            stderr.close()
            diagnostics = log.read_text(encoding="utf-8", errors="replace")
            sys.stderr.write(diagnostics)
    if status:
        raise RuntimeError(f"BoT-SORT cost worker exited with status {status}")
    if costs.dispatches:
        verify_placement(diagnostics, backend, costs.dispatches, expected_device)
    costs.capacities = [(int(rows), int(columns)) for rows, columns in re.findall(r"botsort_cost_capacity=(\d+)x(\d+)", diagnostics)]


def supports_xyxy(tracks, detections):
    # A tracker box has an angle field set to None for axis-aligned geometry.
    return all(getattr(track, "angle", False) is None for group in (tracks, detections) for track in group)


def with_costs(base, worker, *, verify):
    class NativeCosts(base):
        def get_dists(self, tracks, detections):
            if self.args.with_reid or not supports_xyxy(tracks, detections):
                return super().get_dists(tracks, detections)
            if not tracks or not detections:
                return np.empty((len(tracks), len(detections)), dtype=np.float32)
            scores = np.asarray([detection.score for detection in detections]) if self.args.fuse_score else None
            if scores is not None and scores.dtype != np.float32:
                return super().get_dists(tracks, detections)
            matrix = worker.track_distances(tracks, detections, stage="main_and_unconfirmed", scores=scores)
            if verify:
                np.testing.assert_array_equal(matrix, super().get_dists(tracks, detections))
            return matrix

    return NativeCosts


def with_botsort_costs(worker, *, verify):
    from ultralytics import __version__
    from ultralytics.trackers.bot_sort import BOTSORT
    from ultralytics.trackers.utils import matching

    tracker = with_costs(BOTSORT, worker, verify=verify)
    if __version__ != ULTRALYTICS_VERSION:
        return tracker

    def bind_matching(method, stage):
        def iou_distance(tracks, detections):
            if not supports_xyxy(tracks, detections):
                return matching.iou_distance(tracks, detections)
            matrix = worker.track_distances(tracks, detections, stage=stage)
            if verify:
                np.testing.assert_array_equal(matrix, matching.iou_distance(tracks, detections))
            return matrix

        # BYTETracker calls matching directly at these two sites. Bind the replacement
        # in just these functions so other tracker instances keep the module's IoU.
        local_matching = SimpleNamespace(iou_distance=iou_distance, linear_assignment=matching.linear_assignment)
        rebound = FunctionType(method.__code__, dict(method.__globals__, matching=local_matching),
                               method.__name__, method.__defaults__, method.__closure__)
        rebound.__kwdefaults__ = method.__kwdefaults__
        return rebound

    class NativeAssociation(tracker):
        update = bind_matching(BOTSORT.update, "low_confidence")
        remove_duplicate_stracks = staticmethod(bind_matching(BOTSORT.remove_duplicate_stracks, "duplicates"))

    return NativeAssociation
