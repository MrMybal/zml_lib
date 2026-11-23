"""Map FLUX.1 LoRA factors to the fused linear layers of the native denoiser."""

import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


PAIRS = (
    (".lora_down.weight", ".lora_up.weight"),
    (".lora_A.weight", ".lora_B.weight"),
    (".lora_A.default.weight", ".lora_B.default.weight"),
)
DTYPES = {"F32", "F16", "BF16"}
DOUBLE = {
    "attn.to_out.0": "img_attn.proj", "attn.to_add_out": "txt_attn.proj",
    "norm1.linear": "img_mod.lin", "norm1_context.linear": "txt_mod.lin",
    "ff.net.0.proj": "img_mlp.0", "ff.net.2": "img_mlp.2",
    "ff_context.net.0.proj": "txt_mlp.0", "ff_context.net.2": "txt_mlp.2",
}
BASIC = {
    "x_embedder": "img_in", "context_embedder": "txt_in", "proj_out": "final_layer.linear",
    "time_text_embed.timestep_embedder.linear_1": "time_in.in_layer",
    "time_text_embed.timestep_embedder.linear_2": "time_in.out_layer",
    "time_text_embed.text_embedder.linear_1": "vector_in.in_layer",
    "time_text_embed.text_embedder.linear_2": "vector_in.out_layer",
}


def aliases(schema):
    native = {key.removesuffix(".weight"): (index, shape)
              for index, (key, shape) in enumerate(schema.items())
              if key.endswith(".weight") and len(shape) == 2}
    width = native["img_in"][1][0]
    mapped = {}

    def add(alias, name, start=0, rows=None, *, swap_halves=False):
        index, (out_dim, in_dim) = native[name]
        target = (index, start, out_dim if rows is None else rows, in_dim, swap_halves)
        for key in (alias, "diffusion_model." + alias, "transformer." + alias, "base_model.model." + alias,
                    "lora_unet_" + alias.replace(".", "_"), "lora_transformer_" + alias.replace(".", "_")):
            if key in mapped and mapped[key] != target:
                raise ValueError(f"Ambiguous FLUX.1 LoRA name {key}")
            mapped[key] = target

    for name in native:
        add(name, name)
    for alias, name in BASIC.items():
        add(alias, name)
    add("norm_out.linear", "final_layer.adaLN_modulation.1", swap_halves=True)
    for name in native:
        block, _, field = name.rpartition(".img_attn.qkv")
        if block and not field:
            prefix = block.replace("double_blocks.", "transformer_blocks.", 1)
            for part, letter in enumerate("qkv"):
                add(f"{prefix}.attn.to_{letter}", name, part * width, width)
                add(f"{prefix}.attn.add_{letter}_proj", block + ".txt_attn.qkv", part * width, width)
            for alias, native_field in DOUBLE.items():
                add(f"{prefix}.{alias}", f"{block}.{native_field}")
        block, _, field = name.rpartition(".linear1")
        if block.startswith("single_blocks.") and not field:
            prefix = block.replace("single_blocks.", "single_transformer_blocks.", 1)
            for part, letter in enumerate("qkv"):
                add(f"{prefix}.attn.to_{letter}", name, part * width, width)
            add(f"{prefix}.proj_mlp", name, 3 * width, 4 * width)
            add(block + ".linear1_qkv", name, 0, 3 * width)
            add(f"{prefix}.norm.linear", block + ".modulation.lin")
            add(f"{prefix}.proj_out", block + ".linear2")
    return mapped


def factor(checkpoint, key, shape):
    info = checkpoint.get_slice(key)
    if tuple(info.get_shape()) != shape or info.get_dtype() not in DTYPES:
        raise ValueError(f"Invalid LoRA factor {key}, expected a floating tensor of shape {shape}")
    result = checkpoint.get_tensor(key).float().contiguous()
    if not torch.isfinite(result).all().item():
        raise ValueError(f"Nonfinite LoRA factor {key}")
    return result


def add_adapter(path, strength, mapping, records, tensors):
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        keys = set(checkpoint.keys())
        used = set()
        targets = set()
        start = len(records)
        for down, up in PAIRS:
            for key in sorted(key for key in keys if key.endswith(down)):
                alias = key.removesuffix(down)
                if alias not in mapping:
                    raise ValueError(f"{path.name}: unsupported FLUX.1 Schnell LoRA target {alias}")
                up_key = alias + up
                if up_key not in keys:
                    raise ValueError(f"{path.name}: missing LoRA factor {up_key}")
                index, offset, rows, inputs, swap_halves = mapping[alias]
                if (index, offset, rows) in targets:
                    raise ValueError(f"{path.name}: duplicate LoRA target {alias}")
                targets.add((index, offset, rows))
                shape = checkpoint.get_slice(key).get_shape()
                if len(shape) != 2 or not 1 <= shape[0] <= min(rows, inputs) or shape[1] != inputs:
                    raise ValueError(f"{path.name}: invalid LoRA rank or input width for {key}")
                rank = shape[0]
                alpha_key = alias + ".alpha"
                alpha = rank
                if alpha_key in keys:
                    part = checkpoint.get_slice(alpha_key)
                    if tuple(part.get_shape()) not in ((), (1,)):
                        raise ValueError(f"{path.name}: expected scalar {alpha_key}")
                    alpha = checkpoint.get_tensor(alpha_key).item()
                    used.add(alpha_key)
                scale = strength * (alpha / rank)
                if not math.isfinite(scale) or abs(scale) > torch.finfo(torch.float32).max:
                    raise ValueError(f"{path.name}: invalid LoRA scale for {alias}")
                position = len(records)
                tensors[f"down.{position}"] = factor(checkpoint, key, (rank, inputs))
                upper = factor(checkpoint, up_key, (rows, rank))
                # Diffusers orders this modulation as scale, shift. BFL uses shift, scale.
                tensors[f"up.{position}"] = upper.roll(rows // 2, dims=0) if swap_halves else upper
                records.append(dict(index=index, offset=offset, rows=rows, rank=rank, scale=scale))
                used.update((key, up_key))
        if keys - used:
            raise ValueError(f"{path.name}: unsupported LoRA tensor {min(keys - used)}")
        if len(records) == start:
            raise ValueError(f"{path.name}: no supported FLUX.1 LoRA factors")
        return len(records) - start


def prepare_loras(entries, schema, work):
    """Normalize names and factors while leaving the FP8 base checkpoint intact."""
    if not entries:
        return [], []
    mapping = aliases(schema)
    records, tensors, report = [], {}, []
    for file, strength in entries:
        path = Path(file)
        if path.suffix.lower() != ".safetensors" or not path.is_file():
            raise ValueError(f"Choose a local LoRA .safetensors file: {path}")
        if not math.isfinite(strength):
            raise ValueError(f"Invalid LoRA strength for {path.name}")
        count = add_adapter(path, strength, mapping, records, tensors) if strength != 0 else 0
        report.append({"name": path.name, "strength": strength, "modules": count})
    if not records:
        return [], report
    metadata, weights = work / "lora.json", work / "lora.safetensors"
    metadata.write_text(json.dumps(records, allow_nan=False, separators=(",", ":")), encoding="utf-8")
    save_file(tensors, weights)
    return [metadata, weights], report
