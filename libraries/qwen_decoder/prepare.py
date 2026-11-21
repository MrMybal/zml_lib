"""Inspect or merge the local Qwen 0.5B and 1.7B checkpoints into FP32 SafeTensors."""

import argparse
import json
import math
from pathlib import Path
import struct

from safetensors import safe_open


def expected(architecture):
    if architecture == "qwen2":
        count, width, expansion, queries, kv, head = 24, 896, 4864, 14, 2, 64
    else:
        count, width, expansion, queries, kv, head = 28, 2048, 6144, 16, 8, 128
    tensors = {"model.embed_tokens.weight": (151936, width), "model.norm.weight": (width,)}
    if architecture == "qwen3":
        tensors["lm_head.weight"] = (151936, width)
    for index in range(count):
        prefix = f"model.layers.{index}."
        shapes = {
            "input_layernorm.weight": (width,), "post_attention_layernorm.weight": (width,),
            "self_attn.q_proj.weight": (queries * head, width), "self_attn.k_proj.weight": (kv * head, width),
            "self_attn.v_proj.weight": (kv * head, width), "self_attn.o_proj.weight": (width, queries * head),
            "mlp.gate_proj.weight": (expansion, width), "mlp.up_proj.weight": (expansion, width),
            "mlp.down_proj.weight": (width, expansion),
        }
        if architecture == "qwen2":
            shapes.update({"self_attn.q_proj.bias": (queries * head,), "self_attn.k_proj.bias": (kv * head,),
                           "self_attn.v_proj.bias": (kv * head,)})
        else:
            shapes.update({"self_attn.q_norm.weight": (head,), "self_attn.k_norm.weight": (head,)})
        tensors.update((prefix + key, shape) for key, shape in shapes.items())
    return tensors


def inspect_model(directory, profile):
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if (
        config.get("model_type") != profile or config.get("rope_scaling") or config.get("use_sliding_window")
        or any(layer != "full_attention" for layer in (config.get("layer_types") or []))
        or config.get("rope_theta") != 1_000_000 or config.get("hidden_act") != "silu"
        or not math.isclose(config.get("rms_norm_eps", 0), 1e-6, rel_tol=1e-6)
    ):
        raise ValueError(f"Unsupported local configuration for {profile}")
    sources = {}
    layout = {}
    offset = 0
    specs = expected(profile)
    for path in sorted(directory.glob("model*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as shard:
            for name in shard.keys():
                if name not in specs or name in sources:
                    raise ValueError(f"Unexpected or duplicate tensor {name}")
                view = shard.get_slice(name)
                shape = tuple(view.get_shape())
                if shape != specs[name] or view.get_dtype() not in {"BF16", "F16", "F32"}:
                    raise ValueError(f"{name}: expected {specs[name]} as BF16, F16 or F32, found {shape} {view.get_dtype()}")
                size = math.prod(shape) * 4
                layout[name] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + size]}
                offset += size
                sources[name] = path
    if specs.keys() != sources.keys():
        raise ValueError(f"Missing tensors: {sorted(specs.keys() - sources.keys())}")
    return layout, sources, offset


def export(layout, sources, destination):
    header = json.dumps(layout, separators=(",", ":")).encode("utf-8")
    header += b" " * (-len(header) % 8)
    with destination.open("xb") as output:
        try:
            output.write(struct.pack("<Q", len(header)) + header)
            output.flush()
            for shard_path in dict.fromkeys(sources.values()):
                with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
                    for name in shard.keys():
                        if output.tell() != 8 + len(header) + layout[name]["data_offsets"][0]:
                            raise ValueError(f"Unexpected output offset for {name}")
                        shard.get_tensor(name).float().contiguous().numpy().astype("<f4", copy=False).tofile(output)
        except BaseException:
            output.close()
            destination.unlink()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("qwen2", "qwen3"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="omit to check metadata without converting the weights")
    args = parser.parse_args()
    layout, sources, size = inspect_model(args.model.resolve(strict=True), args.profile)
    print(f"{args.profile}: {len(layout)} tensors, {len(set(sources.values()))} shards, {size} bytes as FP32", flush=True)
    if args.output:
        export(layout, sources, args.output.resolve())


if __name__ == "__main__":
    main()
