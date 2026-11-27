"""Compare BiRefNet_HR matte inference against the official FP32 PyTorch graph."""

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForImageSegmentation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.probe_placement import verify_placement


def validate(args, work):
    h, w = args.size
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    with Image.open(args.image) as source:
        rgb = np.asarray(source.convert("RGB").resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32) / 255
    image = np.ascontiguousarray(((rgb - mean) / std).transpose(2, 0, 1)[None])
    source, target = work / "image.f32", work / "matte.f32"
    image.astype("<f4", copy=False).tofile(source)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    net = AutoModelForImageSegmentation.from_pretrained(
        args.original, trust_remote_code=True, local_files_only=True, dtype=torch.float32,
    ).eval().to("cuda")
    with torch.inference_mode():
        expected = net(torch.from_numpy(image).cuda())[-1].sigmoid().cpu().numpy()
    del net
    gc.collect()
    torch.cuda.empty_cache()

    result = subprocess.run(
        [args.probe, "cuda", args.weights, str(h), str(w), source, target, "2"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode:
        sys.stderr.write(result.stderr)
        result.check_returncode()
    native = np.fromfile(target, dtype="<f4").reshape(expected.shape)
    placement = verify_placement(result.stderr, "cuda", 2, args.device, expected_pjrt=args.pjrt)
    delta = native.astype(np.float64) - expected.astype(np.float64)
    error = float(np.sqrt(np.mean(delta**2)))
    if not np.isfinite(error) or error > args.rmse:
        raise RuntimeError(f"Matte RMSE {error:.8g} exceeds {args.rmse}")
    a = native > 0.8
    b = expected > 0.8
    intersection = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return {
        "size": [h, w], "torch_version": torch.__version__, "rmse": error,
        "max_abs": float(np.max(np.abs(delta))), "alpha_mask_iou": float(intersection / union) if union else 1.0,
        "placement": placement,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("probe", "weights", "image", "original", "scratch", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--size", type=int, nargs=2, default=(1024, 1024))
    parser.add_argument("--pjrt", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--rmse", type=float, default=0.001)
    args = parser.parse_args()
    with TemporaryDirectory(prefix="birefnet-", dir=args.scratch) as tmp:
        report = validate(args, Path(tmp))
    with args.report.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
