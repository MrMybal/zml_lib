"""Compare short clips and long-video inference on Windows against PyTorch."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build import ROOT, TEMP_ROOT, run_environment, scratch_directory
from probe_placement import PLATFORMS, verify_placement
from prepare import read_checkpoint
from stream import STRIDE, depths


def clip_rgb(frames, height, width):
    t, y, x = np.indices((frames, height, width))
    return np.stack(((29 * x + 7 * y + 9 * t) % 256, (13 * y + 32 * (x % 2) + 23 * t) % 256,
                     (x * y * 19 + 11 * t) % 256), axis=-1).astype(np.uint8)


def normalize(rgb):
    rgb = rgb.transpose(0, 3, 1, 2).astype(np.float32) / 255
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, :, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, :, None, None]
    return np.ascontiguousarray((rgb - mean) / std)


def check_stream(args, model, probe, weights, work):
    for frames in (1, 22, 23, 24, 32, 44, 45, 69):
        data = normalize(clip_rgb(frames, 14, 14))
        calls = 0

        def predict(batch):
            nonlocal calls
            calls += 1
            return batch[:, 0].clip(0).copy()

        actual = np.stack(list(depths(iter(data), predict)))
        np.testing.assert_allclose(actual, data[:, 0].clip(0), atol=2e-6, rtol=0)
        assert calls == (frames + STRIDE - 1) // STRIDE

    frames, height, width = 51, 28, 42
    rgb = clip_rgb(frames, height, width)
    expected, fps = model.infer_video_depth(rgb, target_fps=15, input_size=height, device="cpu", fp32=True)
    if expected.shape != (frames, height, width) or fps != 15 or not np.isfinite(expected).all():
        raise ValueError("Invalid long-video PyTorch reference")
    if not args.probe:
        print(f"Video Depth Anything PyTorch stream {expected.shape}", flush=True)
        return

    source, output = work / "stream-input.f32le", work / "stream-output.f32le"
    normalize(rgb).astype("<f4", copy=False).tofile(source)
    command = [sys.executable, "-B", str(Path(__file__).with_name("stream.py")),
               "--probe", str(probe), "--weights", str(weights), "--height", str(height), "--width", str(width),
               "--backend", args.backend, "--input", str(source), "--output", str(output)]
    if args.expected_device:
        command += ["--expected-device", args.expected_device]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, errors="replace")
    print(result.stderr, end="", file=sys.stderr)
    result.check_returncode()
    windows = (frames + STRIDE - 1) // STRIDE
    verify_placement(result.stderr, args.backend, windows, args.expected_device)
    actual = np.fromfile(output, dtype="<f4").reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3, equal_nan=False)
    print(f"Video Depth Anything {args.backend} stream frames={frames} windows={windows} "
          f"max_abs={np.max(np.abs(actual - expected)):.9g}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--backend", choices=PLATFORMS, default="cpu")
    parser.add_argument("--expected-device")
    parser.add_argument("--stream", action="store_true", help="also replay 51 frames through long-video inference")
    args = parser.parse_args()
    probe = None
    if args.probe:
        probe = args.probe.resolve(strict=True)
        if not probe.is_relative_to(TEMP_ROOT):
            parser.error("Run Windows validation through scripts/build.py --then")
    sys.path.insert(0, str(args.reference.resolve(strict=True)))
    from video_depth_anything.video_depth import VideoDepthAnything

    torch.set_num_threads(4)
    state = read_checkpoint(args.checkpoint)
    model = VideoDepthAnything(encoder="vits", features=64, out_channels=[48, 96, 192, 384]).eval()
    model.load_state_dict(state, strict=True)
    with scratch_directory() as work:
        os.environ.update(run_environment(work, ROOT.parent / "zml_win" / "bazel-win-tools"))
        weights = work / "video.safetensors"
        if args.probe:
            save_file(state, weights)
        for frames, height, width in ((2, 28, 42), (3, 42, 28)):
            data = normalize(clip_rgb(frames, height, width))
            with torch.inference_mode():
                expected = model(torch.from_numpy(data)[None]).numpy()[0]
            if expected.shape != (frames, height, width) or not np.isfinite(expected).all():
                raise ValueError("Invalid PyTorch reference")
            print(f"PyTorch {frames} frames {height}x{width}: {expected.min():.7g} to {expected.max():.7g}", flush=True)
            if args.probe:
                input_path = work / "input.f32le"
                output_path = work / f"{frames}-frames.f32le"
                data.astype("<f4", copy=False).tofile(input_path)
                command = [str(probe), args.backend, str(weights), str(frames), str(height), str(width), str(input_path), str(output_path)]
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, errors="replace")
                print(result.stderr, end="", file=sys.stderr)
                result.check_returncode()
                verify_placement(result.stderr, args.backend, 1, args.expected_device)
                actual = np.fromfile(output_path, dtype="<f4").reshape(expected.shape)
                np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3, equal_nan=False)
                print(f"Video Depth Anything {args.backend} max_abs={np.max(np.abs(actual - expected)):.9g}", flush=True)
        if args.stream:
            check_stream(args, model, probe, weights, work)


if __name__ == "__main__":
    main()
