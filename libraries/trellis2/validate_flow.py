"""Compare the real checkpoint's TRELLIS.2 token denoiser with FP32 PyTorch."""

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
from safetensors import safe_open
import torch
from torch.nn import functional as F

from prepare import PROFILES
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.probe_placement import verify_placement


@torch.inference_mode()
def reference(original, latent, coords, timestep, context, projection, device, projected):
    # Match the original stored parameter values and compute in FP32.
    # Load one group of Torch matrices at a time.
    with safe_open(original, framework="pt", device="cpu") as weights:
        def fetch(name):
            return weights.get_tensor(name).to(device=device, dtype=torch.float32)

        def linear(name, x):
            return F.linear(x, fetch(name + ".weight"), fetch(name + ".bias"))

        def norm(x, eps=1e-6):
            return F.layer_norm(x, (1536,), eps=eps)

        def split(x):
            return x.reshape(1, -1, 12, 128).transpose(1, 2)

        def heads(proj, name):
            return F.normalize(split(proj), dim=-1) * fetch(name + ".gamma")[None, :, None] * np.sqrt(128)

        def rotate(x, angle):
            value = torch.view_as_complex(x.float().reshape(1, 12, len(coords), 64, 2).contiguous())
            return torch.view_as_real(value * angle).flatten(-2)

        def attend(q, k, v, name):
            value = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(1, -1, 1536)
            return linear(name, value)

        coords = torch.from_numpy(coords).to(device)
        freq = 1 / 10000 ** (torch.arange(21, dtype=torch.float32, device=device) / 21)
        angle = (coords[:, :, None] * freq).flatten(1)
        rotary = torch.polar(torch.ones((len(coords), 64), device=device), F.pad(angle, (0, 1)))
        freq = torch.exp(-np.log(10000) * torch.arange(128, dtype=torch.float32, device=device) / 128)
        time = torch.from_numpy(timestep).to(device)[:, None] * freq
        time = torch.cat((time.cos(), time.sin()), dim=-1)
        time = linear("t_embedder.mlp.2", F.silu(linear("t_embedder.mlp.0", time)))
        time = linear("adaLN_modulation.1", F.silu(time))
        x = linear("input_layer", torch.from_numpy(latent)[None].to(device))
        context = torch.from_numpy(context)[None].to(device)
        projection = torch.from_numpy(projection)[None].to(device)
        for i in range(30):
            scope = f"blocks.{i}"
            shifts = (time + fetch(scope + ".modulation")).reshape(1, 6, 1536)
            shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = shifts.split(1, 1)
            attn = scope + ".self_attn"
            q, k, v = linear(attn + ".to_qkv", norm(x) * (1 + scale_attn) + shift_attn).chunk(3, -1)
            q = rotate(heads(q, attn + ".q_rms_norm"), rotary)
            k = rotate(heads(k, attn + ".k_rms_norm"), rotary)
            x = x + gate_attn * attend(q, k, split(v), attn + ".to_out")

            cross = scope + (".cross_attn.cross_attn_block" if projected else ".cross_attn")
            normal = F.layer_norm(x, (1536,), fetch(scope + ".norm2.weight"), fetch(scope + ".norm2.bias"), 1e-6)
            q = heads(linear(cross + ".to_q", normal), cross + ".q_rms_norm")
            k, v = linear(cross + ".to_kv", context).chunk(2, -1)
            fused = attend(q, heads(k, cross + ".k_rms_norm"), split(v), cross + ".to_out")
            if projected:
                fused = fused + linear(scope + ".cross_attn.proj_linear", projection)
            x = x + fused

            ff = linear(scope + ".mlp.mlp.0", norm(x) * (1 + scale_mlp) + shift_mlp)
            x = x + gate_mlp * linear(scope + ".mlp.mlp.2", F.gelu(ff, approximate="tanh"))
        return linear("out_layer", norm(x, 1e-5))[0].cpu().numpy()


def validate(args, work):
    if args.reference_device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    _, channels, out, projected = PROFILES[args.profile]
    rng = np.random.default_rng(417)
    shape = (args.rows, channels)
    latent = rng.standard_normal(shape, dtype=np.float32)
    grid = 16 if args.rows <= 4096 else 64
    positions = np.arange(args.rows) if grid == 16 else rng.choice(grid**3, size=args.rows, replace=False)
    coords = np.stack(np.unravel_index(positions, (grid, grid, grid)), axis=-1).astype(np.float32)
    step = np.array([735.0], dtype=np.float32)
    context = rng.normal(0, 0.8, (args.context, 1024)).astype(np.float32)
    projection = rng.normal(0, 0.8, (args.rows, max(projected, 1))).astype(np.float32)
    expected = reference(args.original, latent, coords, step, context, projection, args.reference_device, projected)
    gc.collect()
    if args.reference_device.startswith("cuda"):
        torch.cuda.empty_cache()

    paths = [work / f"{i}.f32" for i in range(6)]
    for path, value in zip(paths, (latent, coords, step, context, projection)):
        value.astype("<f4").tofile(path)
    env = os.environ.copy()
    if args.backend == "cuda" and sys.platform == "linux":
        env.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false --xla_gpu_enable_cublaslt=false")
        env.setdefault("NVIDIA_TF32_OVERRIDE", "0")
    reply = subprocess.run(
        [args.probe, args.profile, args.backend, args.prepared, str(args.rows), str(args.context), *paths, "2"],
        text=True, capture_output=True, env=env,
    )
    if reply.returncode:
        raise RuntimeError(reply.stderr[-6000:])
    placement = verify_placement(reply.stderr, args.backend, 2, args.expected_device, expected_pjrt=args.expected_pjrt)
    actual = np.fromfile(paths[-1], dtype="<f4").reshape(args.rows, out)
    delta = actual.astype(np.float64) - expected
    rmse = float(np.sqrt(np.mean(delta ** 2)))
    if not np.isfinite(rmse) or rmse > args.rmse:
        raise ValueError(f"Flow RMSE {rmse:.8g} exceeds {args.rmse}")
    return dict(profile=args.profile, rows=args.rows, context=args.context, output=list(actual.shape),
                rmse=rmse, max_abs=float(np.max(np.abs(delta))), placement=placement,
                reference_device=args.reference_device, torch_version=torch.__version__)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("probe", "original", "prepared", "scratch", "report"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--profile", choices=PROFILES, required=True)
    p.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--expected-pjrt", required=True)
    p.add_argument("--expected-device", required=True)
    p.add_argument("--reference-device", default="cuda:0")
    p.add_argument("--rows", type=int, default=32)
    p.add_argument("--context", type=int, default=20)
    p.add_argument("--rmse", type=float, default=0.001)
    args = p.parse_args()
    row_limit = 4096 if args.profile in ("structure", "structure_multiview") else 65_536
    if not (1 <= args.rows <= row_limit and 1 <= args.context <= 8192):
        p.error("Invalid latent or context size")
    args.scratch = args.scratch.resolve(strict=True)
    if args.scratch.drive.upper() == "C:":
        p.error("--scratch must be outside C:")
    with TemporaryDirectory(prefix="trellis2-flow-", dir=args.scratch) as temp:
        result = validate(args, Path(temp))
    print(json.dumps(result), flush=True)
    with args.report.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
