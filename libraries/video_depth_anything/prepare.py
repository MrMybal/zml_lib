"""Convert a local Video Depth Anything Small checkpoint for the Zig graph."""

import argparse
from pathlib import Path

from safetensors.torch import save_file
import torch


def read_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or len(state) != 351 or any(
        not isinstance(value, torch.Tensor) or value.dtype != torch.float32 for value in state.values()
    ):
        raise ValueError("Expected the 351 FP32 tensors of Video Depth Anything Small")
    return {key: value.detach().contiguous() for key, value in state.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("Choose a new file with an existing parent")
    save_file(read_checkpoint(args.checkpoint.resolve(strict=True)), args.output)


if __name__ == "__main__":
    main()
