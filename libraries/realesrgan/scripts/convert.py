"""Convert a local Real-ESRGAN checkpoint to FP32 SafeTensors."""

import argparse
from pathlib import Path
import tempfile

from safetensors.torch import save_file

from checkpoints import MODELS, read_checkpoint, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix != ".safetensors" or args.checkpoint.resolve() == args.output.resolve():
        parser.error("--output must be a separate .safetensors file")

    tensors, key = read_checkpoint(args.checkpoint, args.model)
    metadata = {
        "format": "pt",
        "model": args.model,
        "checkpoint_key": key,
        "checkpoint_sha256": sha256(args.checkpoint),
        "upstream_checkpoint": MODELS[args.model]["checkpoint"],
        "precision": "float32",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
        staged = Path(temp) / "model.safetensors"
        save_file(tensors, staged, metadata)
        staged.replace(args.output)
    print(f"{args.output} — {len(tensors)} tensors from {key}")


if __name__ == "__main__":
    main()
