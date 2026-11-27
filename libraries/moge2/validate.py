"""Compare native MoGe-2 with the LocalMesh vendored PyTorch implementation."""

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.probe_placement import verify_placement


def compare(actual, expected, threshold):
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    rmse = float(np.sqrt(np.mean(delta**2)))
    if not np.isfinite(rmse) or rmse > threshold:
        raise RuntimeError(f"RMSE {rmse:.8g} exceeds {threshold:.8g}")
    return {"rmse": rmse, "max_abs": float(np.max(np.abs(delta)))}


def validate(args, work):
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    sys.path.insert(0, str(args.upstream.resolve(strict=True)))
    from localmesh_engine._vendu_moge.model.v2 import MoGeModel

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = torch.load(args.original, weights_only=True, map_location="cpu", mmap=True)
    net = MoGeModel(**checkpoint["model_config"])
    net.load_state_dict(checkpoint["model"], strict=True, assign=True)
    del checkpoint
    net.eval().to(args.reference_device)

    h, w = args.size
    with Image.open(args.image) as source:
        image = np.asarray(source.convert("RGB").resize((w, h), Image.Resampling.LANCZOS), dtype=np.float32) / 255
    second = image[:, ::-1].copy()
    rows = round((args.tokens * h / w) ** 0.5)
    cols = round((args.tokens * w / h) ** 0.5)
    images = [np.ascontiguousarray(item.transpose(2, 0, 1)[None]) for item in (image, second)]
    references = []
    with torch.inference_mode():
        for item in images:
            results = net.forward(torch.from_numpy(item).to(args.reference_device), num_tokens=args.tokens)
            keys = ("normal",) if args.profile == "normal" else (
                ("points", "mask", "normal", "metric_scale") if args.profile == "normal-full" else ("points", "mask", "metric_scale")
            )
            references.append({key: results[key].float().cpu().contiguous().numpy() for key in keys})
    del net, results
    gc.collect()
    if args.reference_device.startswith("cuda"):
        torch.cuda.empty_cache()

    paths = []
    for i, item in enumerate(images):
        src = work / f"rgb-{i}.f32"
        dst = work / f"output-{i}.f32"
        item.astype("<f4", copy=False).tofile(src)
        paths.extend((src, dst))
    result = subprocess.run(
        [args.probe, "infer", args.profile, args.backend, args.weights, str(h), str(w), str(rows), str(cols), "2", *paths],
        capture_output=True, text=True, check=False,
    )
    if result.returncode:
        sys.stderr.write(result.stderr)
        result.check_returncode()
    placement = verify_placement(result.stderr, args.backend, 4, args.expected_device, expected_pjrt=args.expected_pjrt)
    cases = []
    for output, expected in zip(paths[1::2], references, strict=True):
        native = np.fromfile(output, dtype="<f4")
        if native.size != sum(value.size for value in expected.values()):
            raise RuntimeError("Unexpected MoGe-2 output length")
        fields = {}
        start = 0
        for key, field in expected.items():
            values = native[start:start + field.size].reshape(field.shape)
            fields[key] = compare(values, field, args.rmse)
            if key == "normal":
                cos = np.sum(values * field, axis=-1)
                fields[key]["angle_mean_degrees"] = float(np.degrees(np.arccos(np.clip(cos, -1, 1))).mean())
            start += field.size
        cases.append(fields)
    return dict(profile=args.profile, size=[h, w], tokens=args.tokens, grid=[rows, cols], threshold=args.rmse,
                placement=placement, reference_device=args.reference_device, torch_version=torch.__version__, cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("probe", "weights", "original", "upstream", "image", "scratch", "report"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--profile", choices=("camera", "normal", "normal-full"), required=True)
    parser.add_argument("--size", type=int, nargs=2, default=(384, 512))
    parser.add_argument("--tokens", type=int, default=3600)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--expected-pjrt", required=True)
    parser.add_argument("--expected-device", required=True)
    parser.add_argument("--reference-device", default="cuda:0")
    parser.add_argument("--rmse", type=float, default=0.001)
    args = parser.parse_args()
    with TemporaryDirectory(prefix="moge2-", dir=args.scratch) as directory:
        report = validate(args, Path(directory))
    with args.report.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
