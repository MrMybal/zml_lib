"""Compare the shared ZML crop with the pinned GVHMR OpenCV and PyTorch code."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build import run_environment, scratch_directory
from probe_placement import PLATFORMS
import crop_io


def fixtures():
    rng = np.random.default_rng(1704)
    boxes = np.array([
        [125.3959536, 169.4925445, 40.0943191],  # A fixed-point rounding tie.
        [25.125, 18.03125, 256.25],            # The crop crosses two image edges.
        [-163.0, -135.0, 9.0],
        [220.0, 74.0, 1e-9],                 # Rounded source corners coincide.
        [600.375, 350.125, 192.123],
        [278.0, 170.0, 1126.39],
        [284.0, 190.0, 1126.41],
        [300.0, 210.0, 2400.0],
        [302.0, 192.0, 4850.0],
    ], dtype=np.float32)
    frames = rng.integers(0, 256, (len(boxes), 188, 317, 3), dtype=np.uint8)
    frames[1, :, :, 0] = 255
    return (
        ("unfiltered", frames[:5], boxes[:5], 0),
        ("mixed", frames, boxes, 3),
        ("narrow", rng.integers(0, 256, (1, 2, 1, 3), dtype=np.uint8),
         np.array([[2400.0 / 510, 2400.0 / 510, 2400.0]], dtype=np.float32), 1),
    )


def load_reference(upstream):
    source = upstream.resolve(strict=True) / "hmr4d/network/hmr2/utils/preproc.py"
    spec = importlib.util.spec_from_file_location("gvhmr_reference_crop", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_crops(frames, boxes, reference):
    import cv2
    import torch

    pixels = []
    for image, box in zip(frames, torch.from_numpy(boxes) * 0.5, strict=True):
        d = (box[2] / 256 / 2).item()
        if d > float(np.float32(1.1)):
            image = cv2.GaussianBlur(image, (5, 5), (d - 1) / 2)
        result, _ = reference.crop_and_resize(image, box[:2], box[2], 256, enlarge_ratio=1.0)
        pixels.append(result[:, 32:224])
    reference_pixels = np.stack(pixels)
    normalized = ((torch.from_numpy(reference_pixels) / 255 - reference.IMAGE_MEAN) /
                  reference.IMAGE_STD).permute(0, 3, 1, 2).numpy()
    return reference_pixels, normalized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--backend", choices=PLATFORMS, default="cuda")
    parser.add_argument("--expected-device")
    args = parser.parse_args()
    probe = args.probe.resolve(strict=True)
    reference = load_reference(args.upstream)

    results = {}
    tools = Path(__file__).resolve().parents[3] / "zml_win" / "bazel-win-tools"
    initial_env = os.environ.copy()
    for name, frames, boxes, expected_antialiased in fixtures():
        with scratch_directory() as work:
            os.environ.update(run_environment(work, tools))
            try:
                reference_pixels, expected = reference_crops(frames, boxes, reference)
                actual, dispatches, antialiased = crop_io.run(
                    probe, args.backend, frames, boxes, work, repeats=2, expected_device=args.expected_device,
                )
                if antialiased != expected_antialiased:
                    raise AssertionError(f"Expected {expected_antialiased} antialiased frames, got {antialiased}")
                recovered = (actual.transpose(0, 2, 3, 1) * reference.IMAGE_STD.numpy() +
                             reference.IMAGE_MEAN.numpy()) * np.float32(255)
                np.testing.assert_array_equal(np.rint(recovered).astype(np.uint8), reference_pixels, err_msg=name)
                np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0, equal_nan=False, err_msg=name)
                results[name] = {"frames": len(frames), "dispatches": dispatches, "antialiased": antialiased,
                                 "max_abs_normalized_error": float(np.max(np.abs(actual - expected)))}
            finally:
                os.environ.clear()
                os.environ.update(initial_env)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
