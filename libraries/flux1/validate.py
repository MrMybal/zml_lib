"""Compare the native FLUX model with a local Black Forest Labs source checkout."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from generate import decode, denoise
from fp8 import FORMAT as SCALED_FP8
from lora import prepare_loras
from prepare import STORAGE, schema_for, write_denoiser

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build import scratch_directory
from probe_placement import PLATFORMS


def scaled_fixture(state, schema, work):
    src, out = work / "scaled-source.safetensors", work / "fixture.safetensors"
    raw, expected = {"scaled_fp8": torch.zeros(2, dtype=torch.float32)}, {}
    for index, (name, weight) in enumerate(state.items()):
        if weight.ndim != 2 or name == "img_in.weight":
            raw[name] = weight.to(torch.bfloat16)
            expected[name] = raw[name].float()
            continue
        layer = name.removesuffix(".weight")
        scale = (0.0075, 0.013, 0.019)[index % 3]
        codes = (weight.float() / scale).to(torch.float8_e4m3fn)
        raw[name] = codes
        expected[name] = codes.float() * scale
        if index % 4 == 0:
            raw[name] = codes.view(torch.uint8)
            raw[layer + ".comfy_quant"] = torch.tensor(list(b'{"format":"float8_e4m3fn"}'), dtype=torch.uint8)
            raw[layer + ".weight_scale"] = torch.tensor([scale], dtype=torch.float32)
        else:
            raw[layer + ".scale_weight"] = torch.tensor(scale, dtype=torch.float32)
        raw[layer + ".input_scale"] = torch.tensor(0.61, dtype=torch.float32)
    save_file({name: value.contiguous() for name, value in raw.items()}, src)
    with safe_open(src, framework="pt", device="cpu") as weights:
        assert write_denoiser(weights, schema, "", out, "auto") == SCALED_FP8
    return out, expected


def make_reference(source_root, profile, storage, checkpoint, schema, work):
    sys.path.insert(0, str(source_root.resolve(strict=True)))
    from flux.model import Flux, FluxParams

    fixture = profile == "fixture"
    config = FluxParams(in_channels=64, out_channels=64, vec_in_dim=768,
                        context_in_dim=32 if fixture else 4096,
                        hidden_size=48 if fixture else 3072, mlp_ratio=4.0,
                        num_heads=3 if fixture else 24, depth=2 if fixture else 19,
                        depth_single_blocks=2 if fixture else 38,
                        axes_dim=[4, 6, 6] if fixture else [16, 56, 56],
                        theta=10_000, qkv_bias=True, guidance_embed=False)
    with torch.device("cpu" if fixture else "meta"):
        torch.manual_seed(183)
        model = Flux(config).eval()
    expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    if expected != schema:
        bad_shapes = [name for name in expected.keys() & schema.keys() if expected[name] != schema[name]]
        raise AssertionError(f"Schema mismatch: missing={expected.keys() - schema.keys()}, "
                             f"extra={schema.keys() - expected.keys()}, shapes={bad_shapes}")
    if fixture and storage == SCALED_FP8:
        checkpoint, converted = scaled_fixture(model.state_dict(), schema, work)
    else:
        if fixture:
            checkpoint = work / "fixture.safetensors"
            dtype = STORAGE[storage][2]
            save_file({name: item.to(dtype).contiguous() for name, item in model.state_dict().items()}, checkpoint)
        with safe_open(checkpoint, framework="pt", device="cpu") as weights:
            scales = weights.get_tensor("weight_scales") if storage == SCALED_FP8 else None
            converted = {}
            for index, name in enumerate(schema):
                value = weights.get_tensor(name)
                converted[name] = (value.float() * scales[index]
                                   if scales is not None and value.dtype == torch.float8_e4m3fn else value.float())
    model.load_state_dict(converted, assign=True)
    return model, checkpoint, config.context_in_dim


@torch.inference_mode()
def reference_cases(model, width):
    rng = np.random.default_rng(82)
    image = rng.standard_normal((2, 3, 64), dtype=np.float32)
    text = rng.standard_normal((7, width), dtype=np.float32) * np.float32(0.4)
    pooled = rng.standard_normal(768, dtype=np.float32) * np.float32(0.5)
    ids = torch.zeros(1, 2, 3, 3)
    ids[0, :, :, 1] = torch.arange(2)[:, None]
    ids[0, :, :, 2] = torch.arange(3)[None, :]
    ids = ids.reshape(1, 6, 3)
    text_ids = torch.zeros(1, len(text), 3)
    original = torch.from_numpy(image.reshape(1, 6, 64))

    def step(tiles, time):
        return model(tiles, ids, torch.from_numpy(text[None]), text_ids,
                     torch.tensor([time], dtype=torch.float32), torch.from_numpy(pooled[None]))

    predicted = step(original, 0.73)
    schedule = np.linspace(1, 0, 5, dtype="<f4")
    sampled = original
    for now, later in zip(schedule[:-1], schedule[1:], strict=True):
        sampled = sampled + float(later - now) * step(sampled, float(now))
    cases = ((np.array([0.73], dtype="<f4"), predicted), (schedule, sampled))
    return image, text, pooled, [(times, tensor.reshape(image.shape).numpy()) for times, tensor in cases]


@torch.no_grad()
def fixture_loras(model, schema, work):
    rng = torch.Generator().manual_seed(384)
    requests = []
    targets = (
        ("diffusion_model.img_in", "img_in.weight", 0, 48, 0.8),
        ("transformer.transformer_blocks.0.attn.to_k", "double_blocks.0.img_attn.qkv.weight", 48, 48, -0.55),
        ("transformer.single_transformer_blocks.0.proj_mlp", "single_blocks.0.linear1.weight", 144, 192, 0.6),
        ("lora_unet_img_in", "img_in.weight", 0, 48, 0.45),
        ("transformer.norm_out.linear", "final_layer.adaLN_modulation.1.weight", 0, 96, 0.75),
    )
    for i, (alias, native, offset, rows, strength) in enumerate(targets):
        rank, alpha = 3, 2.0
        down = torch.randn(rank, schema[native][1], generator=rng) * 0.08
        up = torch.randn(rows, rank, generator=rng) * 0.08
        a, b = (".lora_down.weight", ".lora_up.weight") if i == 3 else (".lora_A.weight", ".lora_B.weight")
        path = work / f"fixture-lora-{i}.safetensors"
        save_file({alias + a: down, alias + b: up, alias + ".alpha": torch.tensor(alpha)}, path)
        requests.append((path, strength))
        update = up @ down
        if native == "final_layer.adaLN_modulation.1.weight":
            scale, shift = update.chunk(2, dim=0)
            update = torch.cat((shift, scale), dim=0)
        model.get_parameter(native)[offset:offset + rows].add_(update, alpha=strength * alpha / rank)
    adapters, report = prepare_loras(requests, schema, work)
    assert [entry["modules"] for entry in report] == [1] * len(targets)
    return adapters


def validate(args, work):
    schema = schema_for(args.probe, args.profile, args.storage)
    model, checkpoint, width = make_reference(args.upstream, args.profile, args.storage, args.weights, schema, work)
    print(f"{args.profile}: checked {len(schema)} tensor shapes, running PyTorch FP32", flush=True)
    adapters = fixture_loras(model, schema, work) if args.fixture_lora else ()
    image, text, pooled, cases = reference_cases(model, width)
    del model
    reports = []
    for times, expected in cases:
        actual, placement = denoise(args.probe, args.backend, args.expected_device, args.profile, args.storage,
                                     checkpoint, image, text, pooled, times, work, adapters)
        delta = np.abs(actual - expected)
        report = {"steps": len(times) - 1, "max_abs": float(delta.max()),
                  "rmse": float(np.sqrt(np.mean(delta ** 2))), "placement": placement}
        print(json.dumps(report, allow_nan=False), flush=True)
        atol, rtol = (6e-5, 6e-5) if args.profile == "fixture" else (2e-3, 2e-3)
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False)
        reports.append(report)
        (work / "flux-out.f32").unlink()
    return reports


@torch.inference_mode()
def validate_vae(args, work):
    from flux.modules.autoencoder import Decoder

    schema = schema_for(args.probe, "vae")
    with torch.device("meta"):
        reference = Decoder(ch=128, out_ch=3, ch_mult=[1, 2, 4, 4], num_res_blocks=2,
                            in_channels=3, resolution=256, z_channels=16).eval()
    expected_shapes = {"decoder." + key: tuple(value.shape) for key, value in reference.state_dict().items()}
    if schema != expected_shapes:
        raise AssertionError("The VAE decoder schema differs from Black Forest Labs")
    with safe_open(args.vae_weights, framework="pt", device="cpu") as storage:
        weights = {key.removeprefix("decoder."): storage.get_tensor(key) for key in schema}
    reference.load_state_dict(weights, assign=True)
    latent = np.random.default_rng(37).standard_normal((16, 4, 6), dtype=np.float32)
    expected = reference(torch.from_numpy(latent[None]) / 0.3611 + 0.1159)[0].numpy()
    actual = decode(args.probe, args.backend, args.expected_device, args.vae_weights, latent, work)
    delta = np.abs(actual - expected)
    print(json.dumps({"stage": "vae", "tensor_count": len(schema), "max_abs": float(delta.max()),
                      "rmse": float(np.sqrt(np.mean(delta ** 2)))}, allow_nan=False), flush=True)
    np.testing.assert_allclose(actual, expected, atol=6e-4, rtol=6e-4, equal_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True, help="directory containing the upstream flux package")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--vae-weights", type=Path, help="also validate the complete FLUX decoder")
    parser.add_argument("--profile", choices=("fixture", "schnell"), default="fixture")
    parser.add_argument("--storage", choices=(*STORAGE, SCALED_FP8), default="e4m3")
    parser.add_argument("--backend", choices=PLATFORMS, default="cpu")
    parser.add_argument("--expected-device")
    parser.add_argument("--fixture-lora", action="store_true", help="Compare split and stacked LoRAs to dense PyTorch patches.")
    args = parser.parse_args()
    if (args.profile == "schnell") != (args.weights is not None):
        parser.error("Supply prepared denoiser weights with --profile schnell")
    if args.fixture_lora and args.profile != "fixture":
        parser.error("--fixture-lora uses the fixture profile")
    args.probe = args.probe.resolve(strict=True)
    torch.set_num_threads(8 if args.weights else 4)
    with scratch_directory() as work:
        validate(args, work)
        if args.vae_weights is not None:
            validate_vae(args, work)


if __name__ == "__main__":
    main()
