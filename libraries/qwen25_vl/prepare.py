"""Prepare the Qwen2.5-VL text or vision encoder from a local ComfyUI SafeTensors file."""

import argparse
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from libraries.flux1.fp8 import FLOAT_DTYPES, read_scale


def read_schema(probe, profile):
    output = subprocess.check_output([probe, profile, "schema"], text=True, encoding="utf-8")
    schema = {}
    for row in output.splitlines():
        name, *dims = row.split("\t")
        if name in schema or not dims:
            raise ValueError(f"Invalid probe schema row: {row}")
        schema[name] = tuple(map(int, dims))
    if not schema:
        raise ValueError("The probe reported an empty schema")
    return schema


def checked_tensors(checkpoint, schema, prefix, dtype):
    keys = set(checkpoint.keys())
    expected = set(schema)
    actual = {name for name in keys if name.startswith(prefix)
              and not name.endswith((".scale_weight", ".scale_input"))}
    if actual != expected:
        raise ValueError(f"Unexpected {prefix} parameters: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    allowed_scales = {name.removesuffix(".weight") + suffix for name in expected
                      if name.endswith(".weight") for suffix in (".scale_weight", ".scale_input")}
    others = {name for name in keys if name.startswith(prefix) and name.endswith((".scale_weight", ".scale_input"))}
    if others - allowed_scales:
        raise ValueError(f"Unexpected {prefix} scale keys: {sorted(others - allowed_scales)}")
    quantized = 0
    tensors = {}
    for name, shape in schema.items():
        header = checkpoint.get_slice(name)
        original = header.get_dtype()
        if tuple(header.get_shape()) != shape or original not in FLOAT_DTYPES | {"F8_E4M3"}:
            raise ValueError(f"Unexpected precision or shape for {name}: {original} {header.get_shape()}, expected {shape}")
        scale_name = name.removesuffix(".weight") + ".scale_weight"
        input_name = name.removesuffix(".weight") + ".scale_input"
        value = checkpoint.get_tensor(name)
        if original == "F8_E4M3":
            if not (name.endswith(".weight") and len(shape) == 2 and scale_name in keys):
                raise ValueError(f"Expected a scaled E4M3FN linear weight for {name}")
            value = value.float() * read_scale(checkpoint, scale_name, weight=True)
            if input_name in keys:
                # The graph applies weight scales and uses floating point activations.
                read_scale(checkpoint, input_name)
            quantized += 1
        elif name.endswith(".weight") and (scale_name in keys or input_name in keys):
            raise ValueError(f"Unexpected FP8 calibration for floating weight {name}")
        value = value.to(dtype).contiguous()
        if not torch.isfinite(value).all().item():
            raise ValueError(f"Nonfinite value after conversion of {name}")
        tensors[name] = value
    return tensors, quantized


def prepare(source, output, schema, component):
    if output.exists() or not output.parent.is_dir():
        raise ValueError("Choose a new file with an existing parent directory")
    with safe_open(source, framework="pt", device="cpu") as checkpoint:
        prefix = "model." if component == "text" else "visual."
        tensors, quantized = checked_tensors(checkpoint, schema, prefix, torch.float16)
        with TemporaryDirectory(prefix="qwen25-vl-", dir=output.parent) as temp:
            staged = Path(temp) / "encoder.safetensors"
            save_file(tensors, staged, metadata={"architecture": "qwen25_vl_7b", "component": component,
                                                "storage": "f16"})
            staged.rename(output)
    return quantized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("probe", "source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--component", choices=("text", "vision"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(6)
    profile = "qwen25_vl_7b" if args.component == "text" else "vision"
    schema = read_schema(args.probe.resolve(strict=True), profile)
    quantized = prepare(args.source.resolve(strict=True), args.output.resolve(), schema, args.component)
    print(f"{len(schema)} FP16 {args.component} tensors written, including {quantized} scaled FP8 matrices", flush=True)


if __name__ == "__main__":
    main()
