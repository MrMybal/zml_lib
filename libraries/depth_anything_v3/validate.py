"""Validate DA3-BASE depth, rays and direct camera head with four LocalMesh example views."""

import argparse
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


def measure(actual, expected, tolerance, name):
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    rmse = float(np.sqrt(np.mean(delta**2)))
    if not np.isfinite(rmse) or rmse > tolerance:
        raise ValueError(f"{name} RMSE {rmse:.8g} exceeds {tolerance}")
    return {"rmse": rmse, "max_abs": float(np.max(np.abs(delta)))}


def validate(args, scratch):
    os.environ["LOCALMESH_ROOT"] = str(args.runtime.resolve(strict=True))
    sys.path.insert(0, str(args.upstream.resolve(strict=True)))
    directory = args.runtime.resolve() / "models/multivue/cameras"
    for folder in (directory / "source/src", directory / "deps"):
        sys.path.insert(0, str(folder))
    from depth_anything_3.utils.io.input_processor import InputProcessor
    from localmesh_engine import config
    from localmesh_engine.multivue.cotes import ROLES, _charger

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    images = []
    for role in ROLES:
        with Image.open(args.photos / f"{role}.png") as source:
            images.append(source.convert("RGB").resize((args.size, args.size), Image.Resampling.LANCZOS))
    stacked, _, _ = InputProcessor()(
        images, process_res=args.size, process_res_method="upper_bound_resize", num_workers=1, sequential=True,
    )
    h, w = stacked.shape[-2:]
    inputs = stacked.reshape(4, 3, h, w).contiguous()
    input_file, output_file = scratch / "input.f32", scratch / "output.f32"
    inputs.numpy().astype("<f4", copy=False).tofile(input_file)
    model = _charger(config)
    with torch.inference_mode():
        features, _ = model.backbone(inputs[None].cuda(), cam_token=None, export_feat_layers=[], ref_view_strategy="first")
        native_ref = model._process_depth_head(features, h, w)
        raw_pose = model.cam_dec(features[-1][1])[0]
        depth_ref = torch.stack((native_ref.depth[0], native_ref.depth_conf[0]), dim=-1).cpu().numpy()
        rays_ref = torch.cat((native_ref.ray[0], native_ref.ray_conf[0, ..., None]), dim=-1).cpu().numpy()
        pose_ref = raw_pose.cpu().numpy()
    del model, features, native_ref
    torch.cuda.empty_cache()

    result = subprocess.run(
        [args.probe, "cuda", args.weights, str(h), str(w), input_file, output_file, "2"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode:
        sys.stderr.write(result.stderr)
        result.check_returncode()
    placement = verify_placement(result.stderr, "cuda", 2, args.expected_device, expected_pjrt=args.expected_pjrt)
    buffer = np.fromfile(output_file, dtype="<f4")
    if buffer.size != depth_ref.size + rays_ref.size + pose_ref.size:
        raise ValueError("Unexpected DA3-BASE output size")
    depth = buffer[:depth_ref.size].reshape(depth_ref.shape)
    rays = buffer[depth_ref.size:depth_ref.size + rays_ref.size].reshape(rays_ref.shape)
    pose = buffer[-pose_ref.size:].reshape(pose_ref.shape)
    return {
        "size": [h, w],
        "native": placement,
        "torch_version": torch.__version__,
        "depth": measure(depth[..., 0], depth_ref[..., 0], args.rmse, "depth"),
        "depth_confidence": measure(depth[..., 1], depth_ref[..., 1], args.rmse, "depth confidence"),
        "rays": measure(rays[..., :6], rays_ref[..., :6], args.rmse, "rays"),
        "ray_confidence": measure(rays[..., 6], rays_ref[..., 6], args.rmse, "ray confidence"),
        "camera_head": measure(pose, pose_ref, args.rmse, "pose"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "report", "probe", "upstream", "photos", "scratch", "runtime"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--size", type=int, default=504)
    parser.add_argument("--expected-device", required=True)
    parser.add_argument("--expected-pjrt", required=True)
    parser.add_argument("--rmse", type=float, default=0.001)
    args = parser.parse_args()
    with TemporaryDirectory(prefix="da3-", dir=args.scratch) as directory:
        report = validate(args, Path(directory))
    with args.report.open("x", encoding="utf-8") as out:
        json.dump(report, out, indent=2)
        out.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
