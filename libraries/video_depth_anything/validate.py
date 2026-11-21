"""Compare short Video Depth Anything Small clips on Windows against PyTorch."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build import ROOT, TEMP_ROOT, scratch_directory


def clip_input(frames, height, width):
    t, y, x = np.indices((frames, height, width))
    rgb = np.stack(((29 * x + 7 * y + 9 * t) % 256, (13 * y + 32 * (x % 2) + 23 * t) % 256,
                    (x * y * 19 + 11 * t) % 256), axis=1).astype(np.float32) / 255
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, :, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, :, None, None]
    return np.ascontiguousarray((rgb - mean) / std)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe", type=Path)
    args = parser.parse_args()
    if args.probe:
        probe = args.probe.resolve(strict=True)
        if not probe.is_relative_to(TEMP_ROOT):
            parser.error("Run Windows validation through scripts/build.py --then")
    sys.path.insert(0, str(args.reference.resolve(strict=True)))
    from video_depth_anything.video_depth import VideoDepthAnything

    torch.set_num_threads(4)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if len(state) != 351 or any(value.dtype != torch.float32 for value in state.values()):
        raise ValueError("Expected the 351 FP32 tensors of Video Depth Anything Small")
    model = VideoDepthAnything(encoder="vits", features=64, out_channels=[48, 96, 192, 384]).eval()
    model.load_state_dict(state, strict=True)
    with scratch_directory() as work:
        weights = work / "video.safetensors"
        if args.probe:
            save_file({key: value.detach().contiguous() for key, value in state.items()}, weights)
        for frames, height, width in ((2, 28, 42), (3, 42, 28)):
            data = clip_input(frames, height, width)
            with torch.inference_mode():
                expected = model(torch.from_numpy(data)[None]).numpy()[0]
            if expected.shape != (frames, height, width) or not np.isfinite(expected).all():
                raise ValueError("Invalid PyTorch reference")
            print(f"PyTorch {frames} frames {height}x{width}: {expected.min():.7g} to {expected.max():.7g}", flush=True)
            if args.probe:
                input_path = work / "input.f32le"
                output_path = work / f"{frames}-frames.f32le"
                data.astype("<f4", copy=False).tofile(input_path)
                command = [str(probe), "cpu", str(weights), str(frames), str(height), str(width), str(input_path), str(output_path)]
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, errors="replace")
                print(result.stderr, end="", file=sys.stderr)
                result.check_returncode()
                if "backend=cpu pjrt=cpu device=cpu:0 shards=1" not in result.stderr:
                    raise ValueError("Probe used an unexpected backend or device")
                actual = np.fromfile(output_path, dtype="<f4").reshape(expected.shape)
                np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3, equal_nan=False)
                print(f"Video Depth Anything cpu:0 max_abs={np.max(np.abs(actual - expected)):.9g}", flush=True)


if __name__ == "__main__":
    main()
