"""Convert official MoGe-2 weights to the native schema from a local model.pt."""

import argparse
from pathlib import Path
import subprocess

from safetensors.torch import save_file
import torch


INTERMEDIATE_LAYERS = [5, 11, 17, 23]
WIDTHS = [1024, 256, 128, 64, 32]
RESAMPLERS = ["conv_transpose", "conv_transpose", "conv_transpose", "bilinear"]


def check_config(config, profile):
    encoder = {"backbone": "dinov2_vitl14", "intermediate_layers": INTERMEDIATE_LAYERS, "dim_out": 1024}
    shared = {
        "dim_res_blocks": WIDTHS, "res_block_in_norm": "none", "res_block_hidden_norm": "none",
        "resamplers": RESAMPLERS,
    }
    expected = dict(
        encoder=encoder,
        neck=dict(
            shared, dim_in=[1026, 2, 2, 2, 2], dim_out=None,
            num_res_blocks=[0, 2, 2, 2, 0],
        ),
        scale_head={"dims": [1024, 1024, 1024, 1]},
        remap_output="exp",
        num_tokens_range=[1200, 3600],
    )
    for head, channels in (("points_head", 3), ("mask_head", 1), ("normal_head", 3)):
        if head == "normal_head" and profile == "camera":
            continue
        expected[head] = dict(
            shared, dim_in=WIDTHS, dim_out=[None, None, None, None, channels],
            num_res_blocks=[0, 1, 1, 1, 0],
        )
    if config != expected:
        raise ValueError("Unsupported MoGe-2 configuration")


def convert(checkpoint, probe, profile, output):
    if output.exists():
        raise FileExistsError(output)
    checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    check_config(checkpoint["model_config"], profile)
    schema = {}
    for line in subprocess.check_output([probe, "schema", profile], text=True).splitlines():
        name, *dims = line.split("\t")
        if name in schema:
            raise ValueError(f"Duplicate schema tensor {name}")
        schema[name] = tuple(map(int, dims))
    source = checkpoint["model"]
    prepared = {}
    for name, shape in schema.items():
        key = name if not name.startswith("pretrained.") else "encoder.backbone." + name[len("pretrained."):]
        tensor = source.pop(key)
        if tensor.dtype != torch.float32 or tensor.shape != shape or not torch.isfinite(tensor).all().item():
            raise ValueError(f"Expected finite FP32 {key} with {shape}, got {tensor.dtype} {list(tensor.shape)}")
        prepared[name] = tensor.contiguous()
    if source:
        raise ValueError(f"Unused checkpoint tensors: {sorted(source)}")
    try:
        save_file(prepared, output, metadata={"source": "MoGe-2", "profile": profile})
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return len(prepared)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("camera", "normal"), required=True)
    args = parser.parse_args()
    count = convert(args.checkpoint, args.probe, args.profile, args.output)
    print(f"{args.profile}: {count} FP32 tensors")


if __name__ == "__main__":
    main()
