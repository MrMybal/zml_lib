"""Compare the four ViT-L/14 activations and relative depth with PyTorch."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import torch
from safetensors.torch import load_file


BLOCKS = (4, 11, 17, 23)
PJRT = {"cpu": "cpu", "cuda": "iree_cuda", "rocm": "iree_rocm", "vulkan": "iree_vulkan"}


def load_reference(source, checkpoint):
    sys.path.insert(0, str(source.resolve(strict=True)))
    from depth_anything_v2.dpt import DepthAnythingV2

    weights = load_file(checkpoint, device="cpu")
    if len(weights) != 407 or any(value.dtype != torch.float16 for value in weights.values()):
        raise ValueError("Expected the 407 FP16 tensors of the ViT-L/14 relative depth checkpoint")
    model = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024]).eval()
    model.load_state_dict(weights, strict=True)
    return model


def input_image(height, width, image_path):
    if image_path:
        from PIL import Image

        with Image.open(image_path) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
        rgb = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
        rgb = torch.nn.functional.interpolate(rgb, (height, width), mode="bilinear", align_corners=False)
    else:
        yy, xx = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width), indexing="ij")
        rgb = torch.stack((xx, yy, (xx + 0.5 * yy).sin())).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    return ((rgb - mean) / std).contiguous()


def reference_outputs(model, image):
    with torch.inference_mode():
        features = model.pretrained.get_intermediate_layers(image, BLOCKS, return_class_token=True)
        depth = model.depth_head(features, image.shape[2] // 14, image.shape[3] // 14).relu().squeeze(1)
    results = {
        f"block_{index}": torch.cat((cls[:, None, :], patches), dim=1).numpy()
        for index, (patches, cls) in zip(BLOCKS, features, strict=True)
    }
    results["depth"] = depth.numpy()
    return results


def run_probe(args, image, mode, repeat, work):
    height, width = image.shape[2:]
    input_path = work / f"{height}x{width}-input.f32le"
    output_path = work / f"{height}x{width}-{mode}.f32le"
    image.numpy().astype("<f4", copy=False).tofile(input_path)
    result = subprocess.run(
        [
            str(args.probe), args.backend, str(args.checkpoint), mode,
            str(height), str(width), str(repeat), str(input_path), str(output_path),
        ],
        capture_output=True, text=True, errors="replace",
    )
    print(result.stderr, end="", file=sys.stderr)
    if result.returncode:
        print(f"Probe {mode} {height}x{width} exited with status {result.returncode:#x}", file=sys.stderr)
    result.check_returncode()
    placements = re.findall(r"backend=(\w+) pjrt=(\w+) device=(.*?) shards=(\d+)", result.stderr)
    if len(placements) != repeat:
        raise ValueError(f"Expected {repeat} placement reports, got {placements}")
    for backend, pjrt, device, shards in placements:
        if (backend, pjrt, shards) != (args.backend, PJRT[args.backend], "1") or not device:
            raise ValueError(f"Unexpected placement {(backend, pjrt, device, shards)}")
        if backend == "cpu" and device != "cpu:0":
            raise ValueError(f"Unexpected CPU device {device}")
    return np.fromfile(output_path, dtype="<f4"), placements[0][2]


def check_outputs(args, expected, values):
    expected_count = sum(value.size for value in expected.values())
    if values.size != expected_count:
        raise ValueError(f"Expected {expected_count} output values, got {values.size}")
    cursor = 0
    errors = {}
    for name, reference in expected.items():
        actual = values[cursor:cursor + reference.size].reshape(reference.shape)
        np.testing.assert_allclose(actual, reference, atol=args.atol, rtol=args.rtol, equal_nan=False, err_msg=name)
        errors[name] = float(np.max(np.abs(actual - reference)))
        cursor += reference.size
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="local checkout containing depth_anything_v2")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe", type=Path, help="omit to run only the PyTorch reference")
    parser.add_argument("--backend", choices=PJRT, default="cpu")
    parser.add_argument("--image", type=Path, help="RGB input image, default is a deterministic color gradient")
    parser.add_argument("--size", nargs=2, type=int, action="append", metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    sizes = args.size or [(28, 42), (42, 28)]
    if len({tuple(size) for size in sizes}) != len(sizes):
        parser.error("Image sizes must be unique")
    if any(h < 14 or w < 14 or h % 14 or w % 14 for h, w in sizes):
        parser.error("Both image dimensions must be positive multiples of 14")
    work = None
    if args.probe:
        test_dir = os.environ.get("ZML_TEST_DIR")
        if not test_dir:
            parser.error("Run the probe through scripts/build.py --then")
        scratch_root = Path(__file__).resolve().parents[2] / "_Temp"
        work = Path(test_dir).resolve(strict=True) / "depth-validation"
        if work.resolve() != work or not work.is_relative_to(scratch_root) or work.parent == scratch_root:
            parser.error("Run the probe through scripts/build.py --then")
        work.mkdir()
    torch.set_num_threads(4)
    model = load_reference(args.reference, args.checkpoint)
    with args.checkpoint.open("rb") as checkpoint:
        report = {"checkpoint_sha256": hashlib.file_digest(checkpoint, "sha256").hexdigest(), "cases": []}
    for height, width in sizes:
        image = input_image(height, width, args.image)
        expected = reference_outputs(model, image)
        depth = expected["depth"]
        if any(not np.isfinite(values).all() for values in expected.values()):
            raise ValueError("Reference returned non-finite values")
        case = {"size": [height, width], "depth_range": [float(depth.min()), float(depth.max())]}
        if args.probe:
            actual, device = run_probe(args, image, "trace", 2, work)
            depth_only, _ = run_probe(args, image, "depth", 1, work)
            case["device"] = device
            case["max_abs_errors"] = check_outputs(args, expected, actual)
            case["reloaded_depth_max_abs_error"] = check_outputs(args, {"depth": depth}, depth_only)["depth"]
        report["cases"].append(case)
    report["torch"] = torch.__version__
    report["precision"] = "FP32 from FP16 checkpoint"
    if args.probe:
        report["backend"] = args.backend
    output = json.dumps(report, indent=2)
    print(output)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as target:
            target.write(output + "\n")


if __name__ == "__main__":
    main()
