"""Compare the LocalMesh DINOv3 conditioning graph with Transformers in FP32."""

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
from safetensors import safe_open
import torch
from torch.nn import functional as F
from transformers import DINOv3ViTModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.probe_placement import verify_placement


def schema(probe, checkpoint):
    lines = subprocess.check_output([probe, "schema"], text=True).splitlines()
    expected = dict((name, tuple(map(int, dims))) for name, *dims in (line.split("\t") for line in lines))
    if len(lines) != len(expected) or len(expected) != 415:
        raise ValueError("Unexpected DINOv3 schema")
    with safe_open(checkpoint, framework="numpy") as weights:
        if set(weights.keys()) != expected.keys():
            raise ValueError("Different tensor names in the DINOv3 checkpoint")
        for name, dims in expected.items():
            tensor = weights.get_slice(name)
            if tuple(tensor.get_shape()) != dims or tensor.get_dtype() != "F32":
                raise ValueError(f"Unexpected shape or dtype for {name}")


def config(model):
    values = json.loads((model / "config.json").read_text(encoding="utf-8"))
    expected = {
        "model_type": "dinov3_vit", "num_hidden_layers": 24, "hidden_size": 1024,
        "intermediate_size": 4096, "num_attention_heads": 16, "num_register_tokens": 4,
        "patch_size": 16, "num_channels": 3, "rope_theta": 100.0,
        "query_bias": True, "key_bias": False, "value_bias": True, "proj_bias": True,
        "mlp_bias": True, "hidden_act": "gelu", "use_gated_mlp": False, "layer_norm_eps": 1e-5,
    }
    for key, value in expected.items():
        if values.get(key) != value:
            raise ValueError(f"Unsupported DINOv3 {key}: {values.get(key)!r}")


def normalize(rgb):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return np.ascontiguousarray(((rgb - mean) / std).transpose(2, 0, 1)[None], dtype="<f4")


@torch.inference_mode()
def references(args, cases):
    if args.reference_device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model = DINOv3ViTModel.from_pretrained(args.model, local_files_only=True, attn_implementation="eager")
    model = model.eval().to(device=args.reference_device, dtype=torch.float32)
    backbone = model if hasattr(model, "layer") else model.model
    outputs = []
    for inputs in cases:
        image = torch.from_numpy(inputs).to(args.reference_device)
        state = model.embeddings(image, bool_masked_pos=None)
        rotary = model.rope_embeddings(image)
        for layer in backbone.layer:
            state = layer(state, position_embeddings=rotary)
        outputs.append(F.layer_norm(state, (1024,)).cpu().numpy())
    del backbone, model
    gc.collect()
    if args.reference_device.startswith("cuda"):
        torch.cuda.empty_cache()
    return outputs


def verify(args, work):
    args.model = args.model.resolve(strict=True)
    config(args.model)
    weights = args.model / "model.safetensors"
    schema(args.probe, weights)
    generator = np.random.default_rng(916)
    cases = [normalize(generator.random((32, 48, 3), dtype=np.float32))]
    if args.image:
        with Image.open(args.image) as source:
            rgb = source.convert("RGB")
            for size in args.size:
                pixels = np.asarray(rgb.resize((size, size), Image.Resampling.LANCZOS), dtype=np.float32) / 255
                cases.append(normalize(pixels))
    expected = references(args, cases)
    env = os.environ.copy()
    if args.backend == "cuda" and sys.platform == "linux":
        env.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false --xla_gpu_enable_cublaslt=false")
        env.setdefault("NVIDIA_TF32_OVERRIDE", "0")
    outputs = []
    for index, (source, reference) in enumerate(zip(cases, expected, strict=True)):
        src, dst, repeat = [work / f"{index}.{label}.f32" for label in ("in", "out", "repeat")]
        source.tofile(src)
        height, width = source.shape[-2:]
        completed = subprocess.run(
            [args.probe, args.backend, weights, str(height), str(width), src, dst, src, repeat],
            capture_output=True, text=True, env=env, check=False,
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr[-6000:])
        placement = verify_placement(completed.stderr, args.backend, 2, args.expected_device, expected_pjrt=args.expected_pjrt)
        actual = np.fromfile(dst, dtype="<f4").reshape(reference.shape)
        np.testing.assert_array_equal(actual, np.fromfile(repeat, dtype="<f4").reshape(reference.shape))
        delta = actual.astype(np.float64) - reference
        rmse = float(np.sqrt(np.mean(delta ** 2)))
        if not np.isfinite(rmse) or rmse > args.rmse:
            raise ValueError(f"Conditioning RMSE {rmse:.8g} exceeds {args.rmse}")
        outputs.append(dict(size=[height, width], shape=list(actual.shape), rmse=rmse,
                            max_abs=float(np.max(np.abs(delta))), placement=placement, repeated=True))
        print(json.dumps(outputs[-1]), flush=True)
    return dict(torch_version=torch.__version__, tensors=415, cases=outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "probe", "scratch"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--size", type=int, nargs="+", default=[512, 1024])
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--expected-pjrt", required=True)
    parser.add_argument("--expected-device", required=True)
    parser.add_argument("--reference-device", default="cuda:0")
    parser.add_argument("--rmse", type=float, default=0.001)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if any(not 16 <= size <= 1024 or size % 16 for size in args.size):
        parser.error("--size accepts multiples of 16 up to 1024")
    args.scratch = args.scratch.resolve(strict=True)
    if args.scratch.drive.upper() == "C:":
        parser.error("--scratch must be outside C:")
    args.probe = args.probe.resolve(strict=True)
    with TemporaryDirectory(prefix="dinov3-", dir=args.scratch) as directory:
        result = verify(args, Path(directory))
    if args.report:
        with args.report.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
