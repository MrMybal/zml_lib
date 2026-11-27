"""Prepare the official local BiRefNet_HR checkpoint for the ZML matte graph."""

import argparse
from pathlib import Path
import subprocess

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


def prepare(args):
    schema = {}
    for line in subprocess.check_output([args.probe, "schema"], text=True).splitlines():
        name, *dims = line.split("\t")
        if name in schema:
            raise ValueError(f"Duplicate schema tensor {name}")
        schema[name] = tuple(map(int, dims))

    tensors = {}
    seen = set()
    if args.output.exists():
        raise FileExistsError(args.output)
    with safe_open(args.checkpoint, framework="numpy") as weights:
        original = set(weights.keys())
        if len(original) != 754:
            raise ValueError("Unexpected official BiRefNet_HR checkpoint tensor count")
        for name, shape in schema.items():
            if name.endswith(".relative_bias"):
                prefix = name.removesuffix(".relative_bias")
                bias_key = prefix + ".relative_position_bias_table"
                index_key = prefix + ".relative_position_index"
                table = weights.get_tensor(bias_key)
                index = weights.get_tensor(index_key)
                if table.shape != (529, shape[0]) or table.dtype != np.float16 or index.shape != (144, 144):
                    raise ValueError(f"Invalid relative position weights for {prefix}")
                if index.dtype != np.int64 or not np.all((0 <= index) & (index < len(table))):
                    raise ValueError(f"Invalid relative position indices for {prefix}")
                value = np.ascontiguousarray(table[index].transpose(2, 0, 1))
                seen.update((bias_key, index_key))
            else:
                value = weights.get_tensor(name)
                seen.add(name)
            if value.shape != shape or value.dtype != np.float16 or not np.isfinite(value).all():
                raise ValueError(f"Expected finite FP16 {name} with shape {shape}")
            tensors[name] = value
        extra = [key for key in original - seen if not key.endswith(".num_batches_tracked")
                 and not key.startswith(("decoder.conv_ms_spvn_", "decoder.gdt_convs_pred_"))]
        if extra:
            raise ValueError(f"Unexpected BiRefNet_HR tensors: {sorted(extra)}")

    try:
        save_file(tensors, args.output, metadata={"source": "ZhengPeng7/BiRefNet_HR"})
    except BaseException:
        args.output.unlink(missing_ok=True)
        raise
    print(f"BiRefNet_HR: {len(tensors)} FP16 tensors for FP32 inference")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("probe", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
