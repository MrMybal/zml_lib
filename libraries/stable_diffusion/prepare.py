"""Extract the three FP32 ZML weight files from a local SD 1.5 checkpoint."""

import argparse
import json
from math import prod
from pathlib import Path
import struct
import subprocess
from tempfile import TemporaryDirectory


MANIFEST = {"profile": "sd15", "storage": "f32"}


def probe_path(bin_dir, library, name):
    return (bin_dir / "libraries" / library / (name + ".exe")).resolve(strict=True)


def native_schema(probe, role):
    rows = subprocess.check_output([probe, role, "schema"], text=True, encoding="utf-8").splitlines()
    schema = {}
    for line in rows:
        name, *shape = line.split("\t")
        if name in schema or not shape:
            raise ValueError(f"Invalid {role} schema entry: {line}")
        schema[name] = tuple(map(int, shape))
    if not schema:
        raise ValueError(f"Empty {role} schema")
    return schema


def clip_config():
    from transformers import CLIPTextConfig

    return CLIPTextConfig(vocab_size=49408, hidden_size=768, intermediate_size=3072,
                          num_hidden_layers=12, num_attention_heads=12, max_position_embeddings=77,
                          hidden_act="quick_gelu", pad_token_id=49407, eos_token_id=49407, bos_token_id=49406)


def extract(checkpoint, prefix, schema, output, include=lambda name: True):
    import torch

    actual = {key.removeprefix(prefix) for key in checkpoint.keys()
              if key.startswith(prefix) and include(key.removeprefix(prefix))}
    if actual != schema.keys():
        raise ValueError(f"Unexpected keys for {output.name}: missing={sorted(schema.keys() - actual)}, "
                         f"extra={sorted(actual - schema.keys())}")
    header = {"__metadata__": {"format": "pt", "storage": "f32"}}
    offset = 0
    for name, shape in schema.items():
        part = checkpoint.get_slice(prefix + name)
        if tuple(part.get_shape()) != shape or part.get_dtype() not in ("F32", "F16", "BF16"):
            raise ValueError(f"Invalid checkpoint tensor {prefix + name}: {part.get_shape()}, {part.get_dtype()}")
        end = offset + prod(shape) * 4
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, end]}
        offset = end
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    with output.open("xb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        for name in schema:
            value = checkpoint.get_tensor(prefix + name).to(torch.float32).contiguous()
            if not torch.isfinite(value).all().item():
                raise ValueError(f"Nonfinite checkpoint tensor {prefix + name}")
            value.numpy().tofile(stream)


def prepare(checkpoint, bin_dir, output):
    from safetensors import safe_open
    import torch
    from transformers import CLIPTextModel

    probe = probe_path(bin_dir, "stable_diffusion", "sd15_probe")
    with torch.device("meta"):
        clip = CLIPTextModel(clip_config())
    clip_schema = {name: tuple(value.shape) for name, value in clip.state_dict().items()}
    if output.exists():
        raise FileExistsError(output)
    with TemporaryDirectory(prefix="sd15-prepare-", dir=output.parent) as temp:
        staged = Path(temp) / "bundle"
        staged.mkdir()
        with safe_open(checkpoint, framework="pt") as source:
            extract(source, "model.diffusion_model.", native_schema(probe, "unet"), staged / "unet.safetensors")
            extract(source, "first_stage_model.", native_schema(probe, "vae"), staged / "vae.safetensors",
                    lambda name: name.startswith(("decoder.", "post_quant_conv.")))
            extract(source, "cond_stage_model.transformer.", clip_schema, staged / "clip.safetensors",
                    lambda name: name != "text_model.embeddings.position_ids")
        (staged / "sd15.json").write_text(json.dumps(MANIFEST) + "\n", encoding="utf-8")
        staged.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--bin-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.checkpoint.resolve(strict=True), args.bin_dir, args.output.resolve())


if __name__ == "__main__":
    main()
