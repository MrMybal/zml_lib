"""Validate the ZML YOLOv8x letterbox against OpenCV and the FP32 launcher."""

import argparse
from contextlib import contextmanager, redirect_stdout
from itertools import islice
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build import ROOT, run_environment, scratch_directory
from probe_placement import PLATFORMS, verify_placement
import yolov8_coco_io as detector


@contextmanager
def test_files():
    with scratch_directory() as work:
        cache = work / "cuda-cache"
        cache.mkdir()
        os.environ.update(run_environment(work, ROOT.parent / "zml_win" / "bazel-win-tools"), CUDA_CACHE_PATH=str(cache))
        yield work


def reference(frame):
    h, w = frame.shape[:2]
    ratio = min(640 / h, 640 / w)
    rw, rh = max(1, round(w * ratio)), max(1, round(h * ratio))
    if (rw, rh) != (w, h):
        frame = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (640 - rw) % 32, (640 - rh) % 32
    left, top = dw // 2, dh // 2
    padded = cv2.copyMakeBorder(frame, top, dh - top, left, dw - left, cv2.BORDER_CONSTANT, value=(114,) * 3)
    return np.ascontiguousarray(padded[..., ::-1].transpose(2, 0, 1))


def compare_pixels(args, frames, work):
    source, output = work / "images.bgr", work / "letterbox.f32"
    count, (h, w) = detector.write_bgr(frames, source)
    shape = detector.letterbox_shape(args.probe, (h, w))
    command = [args.probe, "letterbox", args.backend, str(w), str(h), source, output, str(count), "2"]
    result = subprocess.run(command, stdout=sys.stderr, stderr=subprocess.PIPE, text=True)
    sys.stderr.write(result.stderr)
    result.check_returncode()
    verify_placement(result.stderr, args.backend, 2 * count, args.expected_device)
    channels = 3 * math.prod(shape)
    if output.stat().st_size != count * channels * 4:
        raise ValueError("Unexpected letterbox buffer size")
    worst = 0.0
    with source.open("rb") as originals, output.open("rb") as prepared:
        for index in range(count):
            bgr = np.fromfile(originals, dtype=np.uint8, count=h * w * 3).reshape(h, w, 3)
            actual = np.fromfile(prepared, dtype="<f4", count=channels).reshape(3, *shape)
            pixels = reference(bgr)
            if not np.isfinite(actual).all() or (actual < 0).any() or (actual > 1).any():
                raise ValueError(f"Invalid FP32 letterbox at frame {index}")
            np.testing.assert_array_equal(np.rint(actual * np.float32(255)), pixels, err_msg=f"frame {index}")
            expected = pixels.astype(np.float32) / np.float32(255)
            np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0, err_msg=f"frame {index}")
            worst = max(worst, float(np.max(np.abs(actual - expected))))
    return {"frames": count, "input_hw": [h, w], "canvas_hw": list(shape), "rgb_values": count * channels,
            "native_outputs": count * 2, "max_pixel_error": 0, "max_fp32_error": worst}


def synthetic_weights(probe, path, *, person_bias=0.0):
    from safetensors.numpy import save_file

    schema = subprocess.check_output([probe, "schema", "coco"], text=True)
    rng = np.random.default_rng(61269)
    weights = {}
    for line in schema.splitlines():
        name, *dimensions = line.split("\t")
        shape = tuple(map(int, dimensions))
        if ".dfl." in name:
            values = np.arange(16, dtype=np.float32).reshape(shape)
        elif name.endswith((".bn.weight", ".running_var")):
            values = np.ones(shape, dtype=np.float32)
        elif name.endswith(".running_mean"):
            values = np.zeros(shape, dtype=np.float32)
        else:
            values = rng.standard_normal(shape, dtype=np.float32)
            values *= np.float32(1.5 / math.sqrt(math.prod(shape[1:])) if len(shape) == 4 else 0.03)
        if name.startswith("model.22.cv3.") and name.endswith(".2.bias"):
            values[0] += np.float32(person_bias)
        weights[name] = values
    save_file(weights, path, metadata={"architecture": "yolov8x-coco", "origin": "synthetic"})


def compare_detector(args, rng, work):
    h, w = 481, 853
    frames = rng.integers(0, 256, (2, h, w, 3), dtype=np.uint8)
    weights = work / "synthetic.safetensors"
    synthetic_weights(args.probe, weights)
    bgr, fp32 = work / "input.bgr", work / "input.f32"
    detector.write_bgr(frames, bgr)
    with fp32.open("xb") as stream:
        for frame in frames:
            (reference(frame).astype("<f4") / np.float32(255)).tofile(stream)
    canvas = detector.letterbox_shape(args.probe, (h, w))
    values = []
    for name, source, input_shape in (("bgr", bgr, (h, w)), ("fp32", fp32, None)):
        output = work / f"{name}.f32"
        shape = detector.infer(args.probe, args.backend, weights, source, output, len(frames), canvas, 2,
                               args.expected_device, bgr_shape=input_shape)
        values.append(np.fromfile(output, dtype="<f4").reshape(shape))
    native, expected = values
    if not np.isfinite(native).all() or not np.isfinite(expected).all() or np.array_equal(expected[0], expected[1]):
        raise ValueError("Invalid synthetic predictions or input-independent detector")
    np.testing.assert_allclose(native, expected, atol=2e-3, rtol=2e-5, equal_nan=False)
    return {"frames": len(frames), "canvas_hw": list(canvas), "native_outputs": 8,
            "max_box_error": float(np.abs(native[:, :4] - expected[:, :4]).max()),
            "max_score_error": float(np.abs(native[:, 4:] - expected[:, 4:]).max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--backend", choices=PLATFORMS, default="cuda")
    parser.add_argument("--expected-device")
    parser.add_argument("--video", type=Path, help="also compare a local video frame by frame")
    parser.add_argument("--frames", type=int, default=195)
    parser.add_argument("--model-smoke", action="store_true", help="compare the fused detector using synthetic weights")
    args = parser.parse_args()
    args.probe = args.probe.resolve(strict=True)
    if not 1 <= args.frames <= 4096:
        parser.error("--frames must be between 1 and 4096")
    results = {}
    rng = np.random.default_rng(1771)
    with redirect_stdout(sys.stderr):
        for h, w in ((720, 1280), (481, 853), (713, 401), (378, 640), (61, 31), (1, 640)):
            with test_files() as work:
                report = compare_pixels(args, rng.integers(0, 256, (2, h, w, 3), dtype=np.uint8), work)
                results[f"{w}x{h}"] = report
        if args.video:
            with test_files() as work, detector.read_frames(args.video.resolve(strict=True)) as frames:
                results["video"] = compare_pixels(args, islice(frames, args.frames), work)
                if results["video"]["frames"] != args.frames:
                    raise ValueError(f"Expected {args.frames} decoded video frames")
        if args.model_smoke:
            with test_files() as work:
                results["synthetic_model"] = compare_detector(args, rng, work)
    print(json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()
