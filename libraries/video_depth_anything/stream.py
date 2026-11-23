"""Stream Video Depth Anything Small on consecutive normalized RGB frames."""

import argparse
from contextlib import contextmanager, nullcontext, redirect_stdout
from itertools import count
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build import ROOT, run_environment, scratch_directory
from probe_placement import PLATFORMS, verify_placement


WINDOW = 32
ANCHORS = (0, 12, 24, 25, 26, 27, 28, 29, 30, 31)
OVERLAP = len(ANCHORS)
STRIDE = WINDOW - OVERLAP
BLEND = 8
ALIGN = OVERLAP - BLEND


def align(depth, reference):
    x, y = depth[:ALIGN], reference
    xx = np.sum(x * x)
    x1 = np.sum(x)
    yy = np.sum(y)
    xy = np.sum(x * y)
    n = np.float32(x.size)
    determinant = xx * n - x1 * x1
    if determinant == 0:
        return depth
    scale = (n * xy - x1 * yy) / determinant
    offset = (xx * yy - x1 * xy) / determinant
    return np.maximum(depth * scale + offset, 0)


def depths(frames, predict):
    """Yield one F32 depth map per source frame, retaining one window and its seams."""
    source = iter(frames)
    first = next(source, None)
    if first is None:
        raise ValueError("Expected at least one frame")
    shape = first.shape
    if len(shape) != 3 or shape[0] != 3 or any(not dim or dim % 14 for dim in shape[1:]):
        raise ValueError("Expected normalized RGB [3,H,W] with dimensions divisible by 14")
    batch = np.empty((WINDOW, *shape), dtype="<f4")
    seen = 0
    exhausted = False
    last = first

    def fill(offset):
        nonlocal last, seen, exhausted
        for index in range(offset, WINDOW):
            frame = next(source, None) if not exhausted else None
            if frame is None:
                exhausted = True
            else:
                if frame.shape != shape:
                    raise ValueError("Frame dimensions changed within the video")
                if not np.isfinite(frame).all():
                    raise ValueError("RGB frame contains nonfinite values")
                last = frame
                seen += 1
            batch[index] = last

    def evaluate():
        output = predict(batch)
        if output.shape != (WINDOW, *shape[1:]) or not np.isfinite(output).all():
            raise ValueError("The depth probe returned an invalid window")
        return output

    if not np.isfinite(first).all():
        raise ValueError("The first RGB frame contains nonfinite values")
    batch[0] = first
    seen += 1
    fill(1)
    current = evaluate()
    reference = current[list(ANCHORS[:ALIGN])].copy()
    tail = current[-BLEND:]
    emitted = min(WINDOW - BLEND, seen)
    yield from current[:emitted]

    coefficients = np.linspace(0, 1, BLEND, dtype=np.float32)[:, None, None]
    for start in count(STRIDE, STRIDE):
        if exhausted and start >= seen:
            break
        batch[:OVERLAP] = batch[list(ANCHORS)]
        fill(OVERLAP)
        current = align(evaluate(), reference)
        seam = tail * (1 - coefficients) + current[ALIGN:OVERLAP] * coefficients
        block = np.concatenate((seam, current[OVERLAP:-BLEND]))
        available = min(len(block), seen - emitted)
        yield from block[:available]
        emitted += available
        tail = current[-BLEND:]
        reference[1:] = current[list(ANCHORS[1:ALIGN])]
    yield from tail[:seen - emitted]


@contextmanager
def native_windows(probe, backend, weights, height, width, work, *, expected_device=None, environment=None):
    log = work / "depth-probe.log"
    windows = 0
    with log.open("xb") as diagnostics:
        command = [probe, "windows", backend, weights, str(height), str(width)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=diagnostics, env=environment)

        def predict(batch):
            nonlocal windows
            if batch.shape != (WINDOW, 3, height, width):
                raise ValueError("Invalid RGB inference window")
            process.stdin.write(memoryview(np.ascontiguousarray(batch, dtype="<f4")).cast("B"))
            process.stdin.flush()
            length = WINDOW * height * width * 4
            payload = process.stdout.read(length)
            if len(payload) != length:
                raise RuntimeError("Truncated depth response from the native worker")
            windows += 1
            return np.frombuffer(payload, dtype="<f4").reshape(WINDOW, height, width)

        try:
            yield predict
            process.stdin.close()
            status = process.wait()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdin.close()
            process.stdout.close()
            diagnostics.close()
            report = log.read_text(encoding="utf-8", errors="replace")
            sys.stderr.write(report)
    if status:
        raise RuntimeError(f"Depth worker exited with status {status}")
    verify_placement(report, backend, windows, expected_device)


def read_rgb(source, height, width):
    length = 3 * height * width * 4
    while payload := source.read(length):
        if len(payload) != length:
            raise ValueError("Incomplete RGB frame at the end of the stream")
        yield np.frombuffer(payload, dtype="<f4").reshape(3, height, width)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--backend", choices=PLATFORMS, default="cpu")
    parser.add_argument("--expected-device")
    parser.add_argument("--input", default="-")
    parser.add_argument("--output", default="-")
    args = parser.parse_args()
    if any(dim <= 0 or dim % 14 for dim in (args.height, args.width)):
        parser.error("Height and width must be positive multiples of 14")
    probe, weights = (path.resolve(strict=True) for path in (args.probe, args.weights))
    source = nullcontext(sys.stdin.buffer) if args.input == "-" else open(args.input, "rb")
    with source, redirect_stdout(sys.stderr), scratch_directory() as work:
        tools = ROOT.parent / "zml_win" / "bazel-win-tools"
        env = run_environment(work, tools)
        env["CUDA_CACHE_DISABLE"] = "1"
        output = nullcontext(sys.__stdout__.buffer) if args.output == "-" else open(args.output, "xb")
        with output as sink, native_windows(probe, args.backend, weights, args.height, args.width, work,
                                             expected_device=args.expected_device, environment=env) as predict:
            for depth in depths(read_rgb(source, args.height, args.width), predict):
                sink.write(memoryview(np.ascontiguousarray(depth, dtype="<f4")).cast("B"))
                sink.flush()


if __name__ == "__main__":
    main()
