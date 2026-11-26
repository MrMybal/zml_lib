"""Prepare the standalone CLIP-L encoder or FLUX.1 decoder for ZML."""

import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libraries.stable_diffusion.prepare import extract, native_schema


def prepare(source, probe, output, profile):
    if output.exists() or not output.parent.is_dir():
        raise ValueError("Choose a new output file with an existing parent directory")
    if profile == "clip_l":
        schema = native_schema(probe)

        def includes(name):
            return name != "text_model.embeddings.position_ids"
    else:
        schema = native_schema(probe, "vae")

        def includes(name):
            return name.startswith("decoder.")
    with TemporaryDirectory(prefix="component-", dir=output.parent) as temp:
        staged = Path(temp) / "weights.safetensors"
        with safe_open(source, framework="pt", device="cpu") as weights:
            extract((weights.keys(), weights.get_tensor), "", schema, staged, includes)
        staged.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("clip_l", "flux1_vae"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source.resolve(strict=True), args.probe.resolve(strict=True), args.output.resolve(), args.profile)


if __name__ == "__main__":
    main()
