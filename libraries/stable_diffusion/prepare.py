"""Extract FP32 ZML weights from a local SD 1.5 or SD 1.5 Inpainting checkpoint."""

import argparse
from contextlib import contextmanager
import json
from math import prod
from pathlib import Path
import struct
import subprocess
from tempfile import TemporaryDirectory
from zipfile import is_zipfile


PROFILES = ("sd15", "sd15-inpaint")


def manifest(profile):
    return {"profile": profile, "storage": "f32"}


@contextmanager
def open_checkpoint(path):
    if path.suffix.lower() == ".safetensors":
        from safetensors import safe_open

        with safe_open(path, framework="pt") as source:
            yield source.keys(), source.get_tensor
    elif path.suffix.lower() == ".ckpt":
        import torch

        weights = torch.load(path, map_location="cpu", weights_only=True, mmap=is_zipfile(path))
        if not isinstance(weights, dict):
            raise ValueError("Expected a checkpoint state dictionary")
        state = weights.get("state_dict", weights)
        if not isinstance(state, dict):
            raise ValueError("Expected a state_dict of checkpoint tensors")
        yield state.keys(), state.__getitem__
    else:
        raise ValueError("Expected a local .safetensors or .ckpt checkpoint")


def probe_path(bin_dir, library, name):
    return (bin_dir / "libraries" / library / (name + ".exe")).resolve(strict=True)


def native_schema(probe, *profile):
    role = " ".join(profile) or probe.stem
    rows = subprocess.check_output([probe, *profile, "schema"], text=True, encoding="utf-8").splitlines()
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

    keys, get_tensor = checkpoint
    actual = {key.removeprefix(prefix) for key in keys
              if key.startswith(prefix) and include(key.removeprefix(prefix))}
    if actual != schema.keys():
        raise ValueError(f"Unexpected keys for {output.name}: missing={sorted(schema.keys() - actual)}, "
                         f"extra={sorted(actual - schema.keys())}")
    header = {"__metadata__": {"format": "pt", "storage": "f32"}}
    offset = 0
    for name, shape in schema.items():
        part = get_tensor(prefix + name)
        if (not isinstance(part, torch.Tensor) or tuple(part.shape) != shape
                or part.dtype not in (torch.float32, torch.float16, torch.bfloat16)):
            raise ValueError(f"Invalid shape or precision for checkpoint tensor {prefix + name}; expected {shape}")
        end = offset + prod(shape) * 4
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, end]}
        offset = end
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    with output.open("xb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        for name in schema:
            value = get_tensor(prefix + name).to(torch.float32).contiguous()
            if not torch.isfinite(value).all().item():
                raise ValueError(f"Nonfinite checkpoint tensor {prefix + name}")
            value.numpy().tofile(stream)


def prepare(checkpoint, bin_dir, output, profile="sd15"):
    import torch
    from transformers import CLIPTextModel

    if profile not in PROFILES:
        raise ValueError(f"Unknown profile: {profile}")
    inpaint = profile == "sd15-inpaint"
    probe = probe_path(bin_dir, "stable_diffusion", "sd15_probe")
    with torch.device("meta"):
        clip = CLIPTextModel(clip_config())
    clip_schema = {name: tuple(value.shape) for name, value in clip.state_dict().items()}
    if output.exists():
        raise FileExistsError(output)
    with TemporaryDirectory(prefix="sd15-prepare-", dir=output.parent) as temp:
        staged = Path(temp) / "bundle"
        staged.mkdir()
        with open_checkpoint(checkpoint) as source:
            unet = native_schema(probe, "inpaint_unet" if inpaint else "unet")
            extract(source, "model.diffusion_model.", unet, staged / "unet.safetensors")
            extract(source, "first_stage_model.", native_schema(probe, "vae"), staged / "vae.safetensors",
                    lambda name: name.startswith(("decoder.", "post_quant_conv.")))
            if inpaint:
                extract(source, "first_stage_model.", native_schema(probe, "vae_encode"), staged / "encoder.safetensors",
                        lambda name: name.startswith(("encoder.", "quant_conv.")))
            extract(source, "cond_stage_model.transformer.", clip_schema, staged / "clip.safetensors",
                    lambda name: name != "text_model.embeddings.position_ids")
        (staged / "sd15.json").write_text(json.dumps(manifest(profile)) + "\n", encoding="utf-8")
        staged.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--bin-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile", choices=PROFILES, default="sd15")
    args = parser.parse_args()
    prepare(args.checkpoint.resolve(strict=True), args.bin_dir, args.output.resolve(), args.profile)


if __name__ == "__main__":
    main()
