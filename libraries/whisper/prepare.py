"""Split a local Transformers Whisper checkpoint into the two ZML graphs."""

import argparse
from pathlib import Path
import subprocess

from safetensors import safe_open
from safetensors.torch import save_file
import torch


def schema_for(probe, profile, stage):
    result = subprocess.run([probe, profile, stage, "schema"], check=True, capture_output=True, text=True)
    return {name: tuple(map(int, shape)) for name, *shape in (line.split("\t") for line in result.stdout.splitlines())}


def prepare(probe, profile, source, directory):
    with safe_open(source, framework="pt", device="cpu") as checkpoint:
        if "proj_out.weight" in checkpoint.keys() and not torch.equal(
            checkpoint.get_tensor("proj_out.weight"), checkpoint.get_tensor("model.decoder.embed_tokens.weight")
        ):
            raise ValueError("Expected the tied Whisper token embedding and projection")
        schemas = {stage: schema_for(probe, profile, stage) for stage in ("encoder", "decoder")}
        for schema in schemas.values():
            for name, shape in schema.items():
                part = checkpoint.get_slice(name)
                if tuple(part.get_shape()) != shape or part.get_dtype() not in ("F32", "F16", "BF16"):
                    raise ValueError(f"Unexpected shape or precision for {name}")
        directory.mkdir()
        for stage, schema in schemas.items():
            values = {name: checkpoint.get_tensor(name).float().contiguous() for name in schema}
            save_file(values, directory / f"{stage}.safetensors", metadata={"profile": profile, "precision": "FP32"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--profile", choices=("tiny", "large_v3_turbo"), required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("Choose a new directory with an existing parent")
    prepare(args.probe.resolve(strict=True), args.profile, args.source.resolve(strict=True), args.output)


if __name__ == "__main__":
    main()
