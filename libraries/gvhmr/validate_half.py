"""Compare the ZML half-scale and crop graphs with GVHMR's PyAV reference."""

import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import sys

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build import run_environment, scratch_directory
from probe_placement import PLATFORMS
import crop_io
from validate_crop import load_reference, reference_crops


def reference_half(frames):
    graph = av.filter.Graph()
    result = []
    for i, image in enumerate(frames):
        frame = av.VideoFrame.from_ndarray(image, format="rgb24").reformat(format="gbrp")
        frame.time_base = Fraction(1, 15)
        frame.pts = i
        if i == 0:
            source = graph.add_buffer(template=frame)
            scale = graph.add("scale", "iw*0.5:ih*0.5")
            sink = graph.add("buffersink")
            source.link_to(scale)
            scale.link_to(sink)
            graph.configure()
        graph.push(frame)
        result.append(graph.pull().to_ndarray(format="rgb24"))
    return np.stack(result)


def fixtures():
    rng = np.random.default_rng(1709)
    for name, h, w, count in (
        ("video_size", 720, 1280, 5),
        ("odd", 47, 73, 3),
        ("narrow", 1, 7, 1),
    ):
        frames = rng.integers(0, 256, (count, h, w, 3), dtype=np.uint8)
        frames[0, :, ::2] = (255, 15, 31)
        frames[0, :, 1::2] = (12, 31, 246)
        frames[0, :2, :] = (0, 255, 70)
        boxes = np.tile(np.array([w * 0.48, h * 0.53, max(h, w) * 0.71], dtype=np.float32), (count, 1))
        if name == "odd":
            boxes[-1, 2] = 2400  # Includes the crop's antialiasing after resizing.
        yield name, frames, boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--backend", choices=PLATFORMS, default="cuda")
    parser.add_argument("--expected-device")
    args = parser.parse_args()
    probe = args.probe.resolve(strict=True)
    reference = load_reference(args.upstream)
    tools = Path(__file__).resolve().parents[3] / "zml_win" / "bazel-win-tools"
    environment = os.environ.copy()
    results = {}
    for name, frames, boxes in fixtures():
        with scratch_directory() as work:
            os.environ.update(run_environment(work, tools))
            try:
                actual, reduced, dispatches, antialiased = crop_io.run_half(
                    probe, args.backend, tuple(frames), boxes, work, capture_half=True,
                    repeats=2, expected_device=args.expected_device,
                )
                expected_half = reference_half(frames)
                np.testing.assert_array_equal(reduced, expected_half, err_msg=name)
                _, expected = reference_crops(expected_half, boxes, reference)
                np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0, equal_nan=False, err_msg=name)
                results[name] = {"frames": len(frames), "shape": list(reduced.shape[1:]), "dispatches": dispatches,
                                 "rgb_values": reduced.size, "max_abs_rgb_error": 0, "antialiased": antialiased,
                                 "max_abs_crop_error": float(np.max(np.abs(actual - expected)))}
                if name == "odd":
                    production = work / "production"
                    production.mkdir()
                    crops, _, _, _ = crop_io.run_half(probe, args.backend, frames, boxes, production,
                                                      expected_device=args.expected_device)
                    np.testing.assert_array_equal(crops, actual, err_msg="production")
            finally:
                os.environ.clear()
                os.environ.update(environment)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
