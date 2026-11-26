"""Compare every Qwen3-TTS ZML stage with the same FP16 weights in PyTorch FP32."""

import argparse
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
import torch
from torch.nn import functional as F

from prepare import PROFILES, STAGES, native_env, prepare
from libraries.flux1.prepare import schema_for
from scripts.probe_placement import verify_placement


def linear(weights, prefix, value):
    return F.linear(value, weights[prefix + ".weight"], weights.get(prefix + ".bias"))


def rms(value, gain):
    return value * (value.square().mean(-1, keepdim=True) + 1e-6).rsqrt() * gain


def backbone(weights, x, fixture):
    heads, kv, dim = (4, 2, 8) if fixture else (16, 8, 128)
    n = len(x)
    power = torch.arange(dim // 2, dtype=x.dtype, device=x.device) * (2 / dim)
    angles = torch.outer(torch.arange(n, dtype=x.dtype, device=x.device), 1 / 1_000_000**power)
    angles = torch.cat((angles, angles), dim=-1)
    sine, cosine = angles.sin(), angles.cos()
    blocked = torch.ones((n, n), dtype=torch.bool, device=x.device).triu(1)
    layers = sum(key.endswith(".input_layernorm.weight") for key in weights)

    def rotated(value):
        first, second = value.chunk(2, dim=-1)
        return value * cosine + torch.cat((-second, first), dim=-1) * sine

    for index in range(layers):
        layer = f"model.layers.{index}"
        attention = f"{layer}.self_attn"
        normed = rms(x, weights[f"{layer}.input_layernorm.weight"])
        q = linear(weights, f"{attention}.q_proj", normed).reshape(n, heads, dim)
        k = linear(weights, f"{attention}.k_proj", normed).reshape(n, kv, dim)
        v = linear(weights, f"{attention}.v_proj", normed).reshape(n, kv, dim)
        q = rotated(rms(q, weights[f"{attention}.q_norm.weight"]).transpose(0, 1))
        k = rotated(rms(k, weights[f"{attention}.k_norm.weight"]).transpose(0, 1)).repeat_interleave(heads // kv, 0)
        v = v.transpose(0, 1).repeat_interleave(heads // kv, 0)
        scores = (q @ k.transpose(-2, -1)) * dim**-0.5
        attended = (scores.masked_fill(blocked, -torch.inf).softmax(-1) @ v).transpose(0, 1).reshape(n, heads * dim)
        x = x + linear(weights, f"{attention}.o_proj", attended)
        normalized = rms(x, weights[f"{layer}.post_attention_layernorm.weight"])
        mlp = f"{layer}.mlp"
        gate = F.silu(linear(weights, f"{mlp}.gate_proj", normalized))
        x = x + linear(weights, f"{mlp}.down_proj", gate * linear(weights, f"{mlp}.up_proj", normalized))
    return rms(x, weights["model.norm.weight"])


def reference(stage, weights, data, fixture):
    if stage == "text":
        embedded = F.embedding(data.flatten().long(), weights["text_embedding.weight"])
        return linear(weights, "text_projection.linear_fc2", F.silu(linear(weights, "text_projection.linear_fc1", embedded)))
    if stage == "codec":
        ids = data.flatten().long()
        return torch.stack([weights["codec_embedding.weight"][ids[0]], *(
            weights[f"codec_embedding.{i - 1}.weight"][ids[i]] for i in range(1, 16)
        )])
    if stage == "talker":
        hidden = backbone(weights, data, fixture)
        return torch.cat((hidden, linear(weights, "codec_head", hidden)), -1)
    if "small_to_mtp_projection.weight" in weights:
        data = linear(weights, "small_to_mtp_projection", data)
    hidden = backbone(weights, data, fixture)
    return torch.stack([linear(weights, f"lm_head.{i}", hidden[i + 1]) for i in range(15)])


def check_stage(args, stage, work):
    schema = schema_for(args.probe, args.profile, stage)
    checkpoint = work / "model.safetensors"
    fixture = args.profile == "fixture"
    rng = torch.Generator().manual_seed(2511)
    if fixture:
        weights = {}
        for name, shape in schema.items():
            value = torch.randn(shape, generator=rng) * 0.08
            if name.endswith("norm.weight"):
                value += 1
            weights[name] = value.half()
        save_file(weights, checkpoint)
    else:
        prepare(args.model, args.profile, stage, schema, checkpoint)
    with safe_open(checkpoint, framework="pt", device="cpu") as stream:
        weights = {name: stream.get_tensor(name).to(device=args.reference_device, dtype=torch.float32) for name in schema}
    width = 16 if fixture else PROFILES[args.profile][1]

    command = [args.probe, args.profile, stage, args.backend, checkpoint]
    tests = []
    for repeat in range(2):
        if stage == "text":
            data = torch.tensor([3, 21, 2, 12 + repeat], dtype=torch.float32).view(-1, 1)
        elif stage == "codec":
            data = ((torch.arange(16) + repeat) % 16).float().view(-1, 1)
            data[0] = 20 + repeat if fixture else 2149 + repeat
        else:
            data = torch.randn((16 if stage == "predictor" else 7, width), generator=rng) * 0.2
        source, output = work / f"input-{repeat}.f32", work / f"output-{repeat}.f32"
        data.numpy().astype("<f4", copy=False).tofile(source)
        expected = reference(stage, weights, data.to(args.reference_device), fixture).cpu().numpy()
        tests.append((output, expected))
        if repeat == 0:
            command.append(str(len(data)))
        command.extend((source, output))
    del weights
    if args.reference_device.startswith("cuda"):
        torch.cuda.empty_cache()
    proc = subprocess.run(command, env=native_env(args.backend), text=True, capture_output=True, encoding="utf-8", errors="replace")
    if proc.returncode:
        raise RuntimeError(proc.stderr[-6000:])
    placement = verify_placement(proc.stderr, args.backend, len(tests), args.expected_device, expected_pjrt=args.expected_pjrt)
    for sample, (output, expected) in enumerate(tests):
        actual = np.fromfile(output, dtype="<f4").reshape(expected.shape)
        rmse = float(np.square(actual.astype(np.float64) - expected).mean()**0.5)
        print(json.dumps({"stage": stage, "sample": sample, "shape": list(actual.shape), "rmse": rmse,
                          "max_abs": float(np.max(np.abs(actual - expected))), "placement": placement}), flush=True)
        if not np.isfinite(rmse) or rmse > args.rmse:
            raise ValueError(f"{stage} sample {sample}: RMSE {rmse:.6g} exceeds {args.rmse:.6g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--profile", choices=("fixture", *PROFILES), default="fixture")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--expected-device", required=True)
    parser.add_argument("--expected-pjrt")
    parser.add_argument("--reference-device", default="cpu")
    parser.add_argument("--rmse", type=float, default=0.001)
    args = parser.parse_args()
    if (args.profile == "fixture") == bool(args.model):
        parser.error("--model is required for an official profile and omitted for the fixture")
    if not np.isfinite(args.rmse) or args.rmse <= 0:
        parser.error("--rmse must be finite and positive")
    args.probe = args.probe.resolve(strict=True)
    args.scratch = args.scratch.resolve(strict=True)
    if args.scratch.drive.upper() == "C:":
        parser.error("--scratch must be outside C:")
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.inference_mode():
        for stage in STAGES:
            with TemporaryDirectory(prefix=f"qwen-tts-{stage}-", dir=args.scratch) as temp:
                check_stage(args, stage, Path(temp))


if __name__ == "__main__":
    main()
