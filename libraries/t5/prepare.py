"""Extract a Hugging Face T5 or UMT5 encoder as FP32 or FP16 SafeTensors."""

import argparse
from collections import defaultdict
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess

from safetensors import safe_open
from safetensors.torch import save_file
import torch


def schema_for(probe, profile):
    result = subprocess.run([probe, profile, "schema"], check=True, capture_output=True, text=True)
    return {name: tuple(map(int, shape)) for name, *shape in (line.split("\t") for line in result.stdout.splitlines())}


def prepare(source, output, schema, storage="f32"):
    dtype = {"f32": torch.float32, "f16": torch.float16}[storage]
    with ExitStack() as stack:
        if source.is_dir():
            index = source / "model.safetensors.index.json"
            if index.is_file():
                locations = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            else:
                single = source / "model.safetensors"
                keys = stack.enter_context(safe_open(single, framework="pt", device="cpu")).keys()
                locations = dict.fromkeys(keys, single.name)
        else:
            keys = stack.enter_context(safe_open(source, framework="pt", device="cpu")).keys()
            locations = dict.fromkeys(keys, source.name)
            source = source.parent
        chosen = defaultdict(list)
        for name, shape in schema.items():
            key = name
            if key == "shared.weight" and key not in locations:
                key = "encoder.embed_tokens.weight"
            if key not in locations:
                raise ValueError(f"Missing encoder parameter {name}")
            relative = Path(locations[key])
            if relative.drive or relative.root or ".." in relative.parts:
                raise ValueError(f"Invalid shard path in the index: {relative}")
            shard = (source / relative).resolve(strict=True)
            chosen[shard].append((name, key, shape))

        tensors = {}
        for shard, entries in chosen.items():
            with safe_open(shard, framework="pt", device="cpu") as weights:
                for name, key, shape in entries:
                    part = weights.get_slice(key)
                    if tuple(part.get_shape()) != shape or part.get_dtype() not in ("F32", "F16", "BF16"):
                        raise ValueError(f"Unexpected shape or precision for {name}: {part.get_shape()}, {part.get_dtype()}")
                    value = weights.get_tensor(key).to(dtype).contiguous()
                    if storage == "f16" and part.get_dtype() != "F16" and not torch.isfinite(value).all().item():
                        raise ValueError(f"Parameter {name} is outside the finite FP16 range")
                    tensors[name] = value
        save_file(tensors, output, metadata={"component": "text_encoder",
                                             "precision": "FP16" if storage == "f16" else "FP32"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--profile", choices=("t5_xxl", "umt5_xxl"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--storage", choices=("f32", "f16"), default="f32")
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("Choose a new file with an existing parent")
    prepare(source, args.output, schema_for(args.probe.resolve(strict=True), args.profile), args.storage)


if __name__ == "__main__":
    main()
