"""Compare the ZML person NMS with the torchvision CPU implementation."""

import argparse
from contextlib import redirect_stdout
from itertools import islice
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from probe_placement import PLATFORMS
from validate_letterbox import synthetic_weights, test_files
import yolov8_coco_io as detector


def reference(raw):
    import torch
    from torchvision.ops import nms

    predictions = torch.from_numpy(raw)
    scores, classes = predictions[4:].max(0)
    anchors = torch.where((scores > 0.5) & (classes == 0))[0]
    centers, radius = predictions[:2, anchors].T, predictions[2:4, anchors].T / 2
    boxes = torch.cat((centers - radius, centers + radius), dim=1)
    confidence = scores[anchors]
    chosen = nms(boxes, confidence, 0.7)[:detector.NMS_LIMIT]
    values = torch.cat((boxes, confidence[:, None], torch.zeros_like(confidence[:, None])), dim=1)
    return values[chosen].numpy(), anchors[chosen].numpy()


def source_reference(boxes, canvas, source_shape):
    import torch

    h, w = source_shape
    canvas_h, canvas_w = canvas
    gain = min(canvas_h / h, canvas_w / w)
    pad_x = round((canvas_w - w * gain) / 2 - 0.1)
    pad_y = round((canvas_h - h * gain) / 2 - 0.1)
    projected = torch.from_numpy(boxes.copy())
    projected[:, 0::2] -= pad_x
    projected[:, 1::2] -= pad_y
    projected /= gain
    projected[:, 0::2].clamp_(0, w)
    projected[:, 1::2].clamp_(0, h)
    return projected.numpy()


def compare(rows, raw, projection=None):
    actual, indices = detector.detections(rows, raw.shape[1])
    expected, anchors = reference(raw)
    if projection is not None:
        expected[:, :4] = source_reference(expected[:, :4], *projection)
    np.testing.assert_array_equal(indices, anchors)
    np.testing.assert_array_equal(actual, expected)
    return len(indices)


def run_nms(args, predictions, cells, work, *, source_shape=None):
    source, output = work / "raw.f32", work / "kept.f32"
    count = 0
    with source.open("xb") as data:
        for raw in predictions:
            if raw.shape != (detector.COCO_FIELDS, cells) or raw.dtype != np.float32:
                raise ValueError("Unexpected fixture shape or type")
            raw.astype("<f4", copy=False).tofile(data)
            count += 1
    mode = "nms" if source_shape is None else "nms-source"
    cmd = [args.probe, mode, args.backend, str(cells)]
    if source_shape is not None:
        h, w = source_shape
        cmd.extend((str(w), str(h)))
    cmd.extend((source, output, str(count), "2"))
    detector.run_probe(cmd, args.backend, 2 * count, args.expected_device)
    return source, np.fromfile(output, dtype="<f4").reshape(count, detector.NMS_LIMIT, detector.NMS_FIELDS)


def validate_raw(args, predictions, cells, work, *, source_shape=None):
    source, rows = run_nms(args, predictions, cells, work, source_shape=source_shape)
    projection = None
    if source_shape is not None:
        projection = detector.letterbox_shape(args.probe, source_shape), source_shape
    detections = []
    with source.open("rb") as inputs:
        for row in rows:
            raw = np.fromfile(inputs, dtype="<f4", count=detector.COCO_FIELDS * cells).reshape(detector.COCO_FIELDS, cells)
            detections.append(compare(row, raw, projection))
    report = {"frames": len(rows), "anchors": cells, "native_outputs": 2 * len(rows),
              "kept_total": sum(detections), "kept_max": max(detections)}
    if projection is not None:
        report.update(canvas_hw=list(projection[0]), source_hw=list(source_shape), source_boxes_identical=True)
    return report


def edge_cases():
    raw = np.zeros((84, 21), dtype=np.float32)
    raw[:4, :11] = np.array([
        [5, 5, 10, 10], [5, 5, 7, 10], [5, 5, 7.01, 10], [25, 5, 10, 10], [25, 5, 6.99, 10],
        [45, 5, 10, 10], [45, 5, 10, 10], [65, 5, 10, 10], [85, 5, 10, 10],
        [105, 5, 10, 10], [125, 5, 0, 0],
    ], dtype=np.float32).T
    raw[4, :11] = [0.91, 0.9, 0.89, 0.88, 0.87, 0.82, 0.82, 0.76, 0.72, 0.5, 0.55]
    raw[5, 7] = np.float32(0.76)  # Person wins a tie for best class.
    raw[5, 8] = np.float32(0.92)  # A better non-person score excludes the anchor.
    np.testing.assert_array_equal(reference(raw)[1], [0, 1, 3, 4, 5, 7, 10])
    return raw


def sequence(frames, cells):
    rng = np.random.default_rng(5129)
    for frame in range(frames):
        raw = np.zeros((84, cells), dtype=np.float32)
        ids = rng.choice(cells, 40, replace=False)
        raw[:2, ids] = rng.uniform(20, 550, (2, len(ids)))
        raw[2:4, ids] = rng.uniform(10, 160, (2, len(ids)))
        raw[4, ids] = rng.uniform(0.35, 0.98, len(ids))
        raw[5, ids[::3]] = np.float32(0.89)
        raw[4, frame % cells] = np.float32(0.51)
        yield raw


def projection_cases(canvas):
    h, w = canvas
    raw = np.zeros((84, 21), dtype=np.float32)
    raw[:4, :6] = np.array([
        [w * 0.5, h * 0.5, w * 2, h * 2], [-8, h * 0.15, 32, 10], [w * 0.8, h + 4, 14, 30],
        [w * 0.37 + 0.125, h * 0.59 + 0.25, 10.125, 17.625], [w * 0.21, h * 0.072 + 0.5, 4.5, 7.125],
        [w + 4, h / 2, 5, 11],
    ], dtype=np.float32).T
    raw[4, :6] = [0.98, 0.92, 0.87, 0.83, 0.77, 0.71]
    return raw


def validate_model(args, work):
    if args.video:
        with detector.read_frames(args.video.resolve(strict=True)) as video:
            images = tuple(islice(video, 2))
    else:
        rng = np.random.default_rng(17381)
        images = rng.integers(0, 256, (2, 720, 1280, 3), dtype=np.uint8)
    source, weights = work / "images.bgr", work / "synthetic.safetensors"
    count, bgr_shape = detector.write_bgr(images, source)
    if count != 2:
        raise ValueError("Expected two source images")
    synthetic_weights(args.probe, weights, person_bias=1.5)
    canvas = detector.letterbox_shape(args.probe, bgr_shape)
    audit, kept, raw_only, production = (work / f"{stem}.f32" for stem in ("audit", "kept", "raw-only", "production"))
    compact, cells = detector.infer_nms(args.probe, args.backend, weights, source, kept, count, canvas, 2,
                                        args.expected_device, bgr_shape=bgr_shape, raw_output=audit,
                                        source_coordinates=args.project)
    raw_shape = detector.infer(args.probe, args.backend, weights, source, raw_only, count, canvas, 1,
                               args.expected_device, bgr_shape=bgr_shape)
    detector.infer_nms(args.probe, args.backend, weights, source, production, count, canvas, 1,
                       args.expected_device, bgr_shape=bgr_shape, source_coordinates=args.project)
    raw = np.fromfile(audit, dtype="<f4").reshape(raw_shape)
    trimmed = np.fromfile(kept, dtype="<f4").reshape(compact)
    np.testing.assert_array_equal(raw, np.fromfile(raw_only, dtype="<f4").reshape(raw_shape))
    np.testing.assert_array_equal(trimmed, np.fromfile(production, dtype="<f4").reshape(compact))
    projection = (canvas, bgr_shape) if args.project else None
    kept_counts = [compare(rows, values, projection) for rows, values in zip(trimmed, raw, strict=True)]
    if min(kept_counts) == 0 or cells != raw_shape[-1]:
        raise ValueError("Expected valid person detections in both images")
    return {"frames": count, "anchors": cells, "native_outputs": count * 7, "kept_per_frame": kept_counts,
            "raw_identical": True, "without_raw_output_identical": True, "source_coordinates": args.project}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--backend", choices=PLATFORMS, default="cuda")
    parser.add_argument("--expected-device")
    parser.add_argument("--video", type=Path, help="source for the synthetic-weight detector test")
    parser.add_argument("--model-smoke", action="store_true", help="also validate detection followed by NMS with synthetic weights")
    parser.add_argument("--project", action="store_true", help="also validate the source-coordinate outputs")
    args = parser.parse_args()
    args.probe = args.probe.resolve(strict=True)
    report = {}
    capped = np.zeros((84, 8400), dtype=np.float32)
    capped[0] = np.arange(8400, dtype=np.float32) * 20
    capped[2:4] = 8
    capped[4] = np.float32(0.8)
    empty = np.zeros((84, 1), dtype=np.float32)
    single = empty.copy()
    single[:5, 0] = [20, 15, 12, 10, 0.8]
    with redirect_stdout(sys.stderr), test_files() as suite:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(suite / "torchinductor")
        import torch

        torch.set_num_threads(4)
        for name, samples, cells in (("single", [empty, single], 1),
                                     ("edges", [edge_cases()], 21), ("cap", [capped], 8400),
                                     ("sequence", sequence(195, 5040), 5040)):
            with test_files() as work:
                report[name] = validate_raw(args, samples, cells, work)
        with test_files() as work:
            invalid = np.zeros((2, 84, 21), dtype=np.float32)
            invalid[0, 0, 0] = np.nan
            invalid[1, 83, 20] = np.inf
            modes = [("nms", [])]
            if args.project:
                modes.append(("nms-source", ["1280", "720"]))
            for index, frame in enumerate(invalid):
                source, output = work / f"invalid-{index}.f32", work / f"discarded-{index}.f32"
                frame.tofile(source)
                for mode, dimensions in modes:
                    command = [args.probe, mode, args.backend, "21", *dimensions, source, output]
                    result = subprocess.run(command, capture_output=True, text=True)
                    if not result.returncode or "NonFiniteNmsPredictions" not in result.stderr or output.exists():
                        raise AssertionError(f"Expected a native error for {mode} fixture {index}: {result.stderr}")
            report["nonfinite"] = {"frames": 2, "rejected": True}
        if args.project:
            projected = {}
            for h, w in ((720, 1280), (481, 853), (713, 401), (40, 60), (1, 8192)):
                canvas = detector.letterbox_shape(args.probe, (h, w))
                with test_files() as work:
                    projected[f"{w}x{h}"] = validate_raw(args, [projection_cases(canvas)], 21, work, source_shape=(h, w))
            with test_files() as work:
                projected["sequence"] = validate_raw(args, sequence(195, 5040), 5040, work, source_shape=(720, 1280))
            report["projection"] = projected
        if args.model_smoke:
            with test_files() as work:
                report["synthetic_model"] = validate_model(args, work)
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
