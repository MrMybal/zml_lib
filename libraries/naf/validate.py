"""Compare the LocalMesh NAF encoder and sampled features with PyTorch."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import types

import numpy as np
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.probe_placement import verify_placement


def schema(probe, checkpoint):
    lines = subprocess.check_output([probe, "schema"], text=True).splitlines()
    expected = dict((name, tuple(map(int, dims))) for name, *dims in (line.split("\t") for line in lines))
    if len(lines) != len(expected) or len(expected) != 37:
        raise ValueError("Unexpected NAF schema")
    with safe_open(checkpoint, framework="numpy") as weights:
        if set(weights.keys()) != expected.keys():
            raise ValueError("Different tensor names in the NAF checkpoint")
        for name, dims in expected.items():
            tensor = weights.get_slice(name)
            if tuple(tensor.get_shape()) != dims or tensor.get_dtype() != "F32":
                raise ValueError(f"Unexpected shape or dtype for {name}")


def load_reference(args):
    sys.path.insert(0, str(args.upstream.resolve(strict=True)))
    # Only the image encoder and LocalMesh's sampled attention run here. The
    # dense NATTEN kernel is deliberately unavailable to these references.
    natten = types.ModuleType("natten")

    def unavailable(*args, **kwargs):
        raise RuntimeError("The reference unexpectedly called dense NATTEN")

    natten.na2d = unavailable
    sys.modules["natten"] = natten
    from localmesh_engine.multivue.champ.model.naf import NAF
    from localmesh_engine.multivue.naf_echantillon import sample_naf_features
    encoder = NAF().eval()
    encoder.load_state_dict(load_file(args.weights), strict=True)
    return encoder.image_encoder.to(args.reference_device), sample_naf_features


def run(args, prefix, calls):
    env = os.environ.copy()
    if args.backend == "cuda" and sys.platform == "linux":
        env.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false --xla_gpu_enable_cublaslt=false")
        env.setdefault("NVIDIA_TF32_OVERRIDE", "0")
    done = subprocess.run([args.probe, *prefix], text=True, capture_output=True, env=env)
    if done.returncode:
        raise RuntimeError(done.stderr[-6000:])
    return verify_placement(done.stderr, args.backend, calls, args.expected_device, expected_pjrt=args.expected_pjrt)


def compare(label, expected, actual, limit):
    delta = actual.astype(np.float64) - expected
    rmse = float(np.sqrt(np.mean(np.square(delta))))
    if not np.isfinite(rmse) or rmse > limit:
        raise ValueError(f"{label} RMSE {rmse:.8g} exceeds {limit}")
    return dict(rmse=rmse, max_abs=float(np.max(np.abs(delta))))


@torch.inference_mode()
def verify(args, work):
    schema(args.probe, args.weights)
    if args.reference_device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    encoder, sample = load_reference(args)
    rng = np.random.default_rng(316)
    cases = [(rng.random((1, 3, 40, 48), dtype=np.float32), 1),
             (rng.random((1, 3, 40, 48), dtype=np.float32), 2)]
    if args.image:
        with Image.open(args.image) as source:
            rgb = source.convert("RGB")
            for size in args.size:
                pixels = np.asarray(rgb.resize((size, size), Image.Resampling.LANCZOS), dtype=np.float32) / 255
                image = np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])
                cases.append((image, size // 512))
                if args.full_resolution and size == 1024:
                    cases.append((image, 1))
    reports = []
    for index, (image, stride) in enumerate(cases):
        height, width = image.shape[-2:]
        q_size = (height // stride, width // stride)
        q_ref = encoder(torch.from_numpy(image).to(args.reference_device), output_size=q_size)
        src, dst, features_file, points_file, sample_file = [
            work / f"{index}.{name}.f32" for name in ("image", "queries", "features", "points", "samples")
        ]
        image.astype("<f4").tofile(src)
        op = "encode" if stride == 1 else "encode-half"
        placement = run(args, [op, args.backend, args.weights, str(height), str(width), src, dst, "2"], 2)
        native_queries = np.fromfile(dst, dtype="<f4").reshape(1, 256, *q_size)
        encoding = compare("NAF query map", q_ref.cpu().numpy(), native_queries, args.rmse)
        rows, cols = (height // 16, width // 16) if min(q_size) >= 512 else (10, 12)
        depth = 1024 if min(q_size) >= 512 else 12
        feat = rng.normal(size=(1, depth, rows, cols)).astype("<f4")
        corners = np.array([[-1, -1], [1, 1], [-1.1, 1.1], [1, -1], [0, 0], [0.271, -0.853]])
        positions = np.concatenate((corners, rng.uniform(-1, 1, size=(125, 2)))).astype("<f4")
        feat.tofile(features_file)
        positions.tofile(points_file)
        sample_placement = run(args, ["sample", args.backend, *map(str, (*q_size, rows, cols, depth, len(positions))),
                                    dst, features_file, points_file, sample_file], 2)
        reference = sample(q_ref, torch.from_numpy(feat).to(args.reference_device),
                           torch.from_numpy(positions).to(args.reference_device)).cpu().numpy()
        value = np.fromfile(sample_file, dtype="<f4").reshape(len(positions), depth)
        sampling = compare("NAF combined encoding and neighborhood sampling", reference, value, args.rmse)
        report = dict(image=[height, width], queries=list(q_size), feature_shape=[depth, rows, cols],
                      points=len(positions), encoding=encoding, sampling=sampling,
                      encoder_placement=placement, sample_placement=sample_placement)
        reports.append(report)
        print(json.dumps(report), flush=True)
    return dict(torch_version=torch.__version__, tensors=37, cases=reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "probe", "upstream", "scratch"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--size", type=int, nargs="+", choices=(512, 1024), default=[512, 1024])
    parser.add_argument("--full-resolution", action="store_true", help="also sample the 1024-pixel Pixal3D query map")
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--expected-pjrt", required=True)
    parser.add_argument("--expected-device", required=True)
    parser.add_argument("--reference-device", default="cuda:0")
    parser.add_argument("--rmse", type=float, default=0.001)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    args.scratch = args.scratch.resolve(strict=True)
    if args.scratch.drive.upper() == "C:":
        parser.error("--scratch must be outside C:")
    args.probe = args.probe.resolve(strict=True)
    with TemporaryDirectory(prefix="naf-", dir=args.scratch) as directory:
        result = verify(args, Path(directory))
    if args.report:
        with args.report.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
