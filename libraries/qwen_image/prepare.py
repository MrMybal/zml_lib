"""Prepare local ComfyUI Qwen Image Edit 2511 denoiser or single-image VAE weights."""

import argparse
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from libraries.flux1 import fp8
from libraries.flux1.prepare import write_tensors


TEMPORAL_LAYERS = {
    "encoder.downsamples.5.time_conv": (192, 192),
    "encoder.downsamples.8.time_conv": (384, 384),
    "decoder.upsamples.3.time_conv": (768, 384),
    "decoder.upsamples.7.time_conv": (768, 384),
}


def schema_for(probe, profile):
    columns = subprocess.check_output([probe, profile, "schema"], text=True, encoding="utf-8").splitlines()
    schema = {}
    for row in columns:
        name, *dims = row.split("\t")
        if not dims or name in schema:
            raise ValueError(f"Invalid schema row: {row}")
        schema[name] = tuple(map(int, dims))
    if not schema:
        raise ValueError("The probe reported an empty schema")
    return schema


def require_shape(checkpoint, name, shape):
    info = checkpoint.get_slice(name)
    if tuple(info.get_shape()) != shape or info.get_dtype() not in fp8.FLOAT_DTYPES:
        raise ValueError(f"Invalid tensor {name}: expected floating {shape}, found {info.get_dtype()} {info.get_shape()}")


def write_autoencoder(checkpoint, schema, stage, profile):
    prefix = "encoder." if profile == "vae_encode" else "decoder."
    quant_prefix = "conv1." if profile == "vae_encode" else "conv2."
    time = {key + suffix: dimensions for key, (out, in_) in TEMPORAL_LAYERS.items()
            if key.startswith(prefix) for suffix, dimensions in ((".weight", (out, in_, 3, 1, 1)), (".bias", (out,)))}
    actual = {key for key in checkpoint.keys() if key.startswith((prefix, quant_prefix))}
    expected = schema.keys() | time.keys()
    if actual != expected:
        raise ValueError(f"Invalid VAE keys: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    for name, shape in time.items():
        require_shape(checkpoint, name, shape)

    def expected_shape(name, shape):
        if name.endswith(".gamma"):
            dimensions = 2 if ".middle.1." in name else 3
            return (*shape, *([1] * dimensions))
        if len(shape) == 4 and ".resample." not in name and ".middle.1." not in name:
            return (*shape[:2], shape[-1], *shape[-2:])
        return shape

    def values():
        for name, shape in schema.items():
            raw = expected_shape(name, shape)
            require_shape(checkpoint, name, raw)
            tensor = checkpoint.get_tensor(name)
            if len(raw) == 5:
                # In the first frame of a causal convolution only the latest temporal coefficient sees pixels.
                tensor = tensor[:, :, -1, :, :]
            tensor = tensor.reshape(shape).float().contiguous()
            if not torch.isfinite(tensor).all().item():
                raise ValueError(f"Nonfinite VAE parameter {name}")
            yield tensor

    write_tensors(stage, "f32", {name: (shape, "f32") for name, shape in schema.items()}, values())


def write_denoiser(checkpoint, schema, stage):
    keys = set(checkpoint.keys())
    marker = "__index_timestep_zero__"
    require_shape(checkpoint, marker, (0,))
    fields, sources, scales = fp8.plan(checkpoint, schema, "")
    quantized = {source.removesuffix(".weight") for name, source in sources.items() if fields[name][1] == "e4m3"}
    markers = {layer + "." + key for layer in quantized for key in ("input_scale", "weight_scale", "comfy_quant")}
    expected = set(sources.values()) | markers | {marker}
    if keys != expected:
        raise ValueError(f"Invalid Edit 2511 keys: missing={sorted(expected - keys)}, extra={sorted(keys - expected)}")
    fields["weight_scales"] = ((len(schema),), "f32")

    def values():
        for name, source in sources.items():
            value = fp8.tensor_value(checkpoint, source, fields[name][1])
            if fields[name][1] == "e4m3" and torch.any((value & 0x7f) == 0x7f).item():
                raise ValueError(f"Nonfinite FP8 matrix {name}")
            yield value
        yield torch.tensor(scales, dtype=torch.float32)

    write_tensors(stage, "e4m3_scaled", fields, values())
    return len(quantized)


def prepare(source, output, schema, profile):
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ValueError("Choose a new output file in an existing directory")
    with TemporaryDirectory(prefix="qwen-image-", dir=output.parent) as temp:
        staged = Path(temp) / "weights.safetensors"
        quantized = 0
        with safe_open(source, framework="pt", device="cpu") as checkpoint:
            if profile == "edit_2511":
                quantized = write_denoiser(checkpoint, schema, staged)
            else:
                write_autoencoder(checkpoint, schema, staged, profile)
        staged.rename(output)
    return quantized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("vae_encode", "vae_decode", "edit_2511"), required=True)
    for key in ("probe", "source", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    schema = schema_for(args.probe.resolve(strict=True), args.profile)
    converted = prepare(args.source.resolve(strict=True), args.output.resolve(), schema, args.profile)
    print(f"{args.profile}: {len(schema)} tensors, {converted} scaled FP8 matrices", flush=True)


if __name__ == "__main__":
    main()
