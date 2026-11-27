"""Extract the official DA3-BASE four-view inference weights from a local SafeTensors file."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


def prepare(args):
    profile = json.loads(args.config.read_text(encoding="utf-8"))
    config = profile["config"]
    net = config["net"]
    head = config["head"]
    if (profile["model_name"] != "da3-base" or net["name"] != "vitb"
            or net["out_layers"] != [5, 7, 9, 11] or not net["cat_token"]
            or any(net[key] != 4 for key in ("alt_start", "qknorm_start", "rope_start"))
            or (head["dim_in"], head["output_dim"], head["features"], head["out_channels"])
            != (1536, 2, 128, [96, 192, 384, 768])
            or config["cam_dec"]["dim_in"] != 1536):
        raise ValueError("Expected the DA3-BASE configuration distributed by LocalMesh")
    schema = {}
    for line in subprocess.check_output([args.probe, "schema"], text=True).splitlines():
        name, *dims = line.split("\t")
        if name in schema:
            raise ValueError(f"Duplicate schema tensor {name}")
        schema[name] = tuple(map(int, dims))
    if args.output.exists():
        raise FileExistsError(args.output)
    prepared = {}
    with safe_open(args.checkpoint, framework="numpy") as source:
        keys = set(source.keys())
        valid_extras = set()
        for key in keys - schema.keys():
            if key.startswith("model.cam_enc."):
                valid_extras.add(key)
            for suffix in ("output_conv1_aux", "output_conv2_aux"):
                if key.startswith(tuple(f"model.head.scratch.{suffix}.{level}." for level in range(3))):
                    valid_extras.add(key)
        if len(keys) != 437 or keys != schema.keys() | valid_extras:
            raise ValueError(f"Unexpected model keys: {sorted(keys ^ (schema.keys() | valid_extras))}")
        for name, dims in schema.items():
            value = source.get_tensor(name)
            if value.dtype != np.float32 or value.shape != dims or not np.isfinite(value).all():
                raise ValueError(f"Expected finite FP32 {name} of shape {dims}")
            prepared[name] = value
    try:
        save_file(prepared, args.output, metadata={"profile": "DA3-BASE LocalMesh four-view first reference"})
    except BaseException:
        args.output.unlink(missing_ok=True)
        raise
    print(f"DA3-BASE: {len(prepared)} FP32 tensors")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("probe", "config", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
