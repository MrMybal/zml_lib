"""Convert a local SIGA24 checkpoint or run the GVHMR temporal network."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
from zipfile import is_zipfile

import numpy as np
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build import ROOT, run_environment, scratch_directory
from probe_placement import PLATFORMS, verify_placement


FORK_TOOLS = ROOT.parent / "zml_win" / "bazel-win-tools"
PREFIX = "pipeline.denoiser3d."
CONDITION_SHAPES = {
    "obs": (17, 3),
    "f_cliffcam": (3,),
    "f_cam_angvel": (6,),
    "f_imgseq": (1024,),
}


def schema_for(probe):
    stdout = subprocess.check_output([probe, "schema"], text=True)
    schema = {}
    for line in stdout.splitlines():
        name, *dims = line.split("\t")
        if name in schema:
            raise ValueError(f"Duplicate GVHMR weight {name}")
        schema[name] = tuple(map(int, dims))
    if len(schema) != 247:
        raise ValueError("Expected 247 tensor definitions for the SIGA24 temporal model")
    return schema


def load_checkpoint(path, schema):
    import torch

    loaded = torch.load(path, map_location="cpu", weights_only=True, mmap=is_zipfile(path))
    if not isinstance(loaded, dict) or not isinstance(loaded.get("state_dict"), dict):
        raise ValueError("Expected the SIGA24 Lightning checkpoint")
    weights = {key[len(PREFIX):]: value for key, value in loaded["state_dict"].items() if key.startswith(PREFIX)}
    if weights.keys() != schema.keys():
        raise ValueError(f"Unexpected or missing GVHMR weight keys: {sorted(weights.keys() ^ schema.keys())}")
    for name, shape in schema.items():
        value = weights[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != torch.float32:
            raise ValueError(f"Expected {name} with shape {shape} in F32")
    return {name: weight.contiguous() for name, weight in weights.items()}


def save_weights(weights, path):
    save_file(weights, path, metadata={"architecture": "GVHMR SIGA24 temporal", "storage": "F32"})


def pack(conditions):
    if conditions.keys() != CONDITION_SHAPES.keys():
        raise ValueError(f"Expected four conditions: {', '.join(CONDITION_SHAPES)}")
    obs = conditions["obs"]
    if obs.ndim != 3 or not 1 <= len(obs) <= 4096:
        raise ValueError("Expected a single sequence of 1–4096 frames")
    packed = []
    for name, shape in CONDITION_SHAPES.items():
        value = conditions[name]
        if value.shape != (len(obs), *shape) or value.dtype.kind != "f":
            raise ValueError(f"Invalid shape or dtype for {name}")
        narrowed = np.asarray(value, dtype="<f4")
        if not np.isfinite(narrowed).all():
            raise ValueError(f"Non-finite FP32 condition {name}")
        packed.append(narrowed.reshape(len(obs), -1))
    return np.concatenate(packed, axis=1)


def read_conditions(path):
    data = np.load(path, allow_pickle=False)
    if isinstance(data, np.ndarray):
        raise ValueError("Expected a conditions .npz archive")
    with data:
        return {name: data[name] for name in data.files}


def run(probe, backend, weights, conditions, work, stem, repeats=1, *, expected_device=None):
    values = pack(conditions)
    source, target = work / f"{stem}.input.f32", work / f"{stem}.output.f32"
    values.tofile(source)
    result = subprocess.run([probe, backend, weights, str(len(values)), source, target, str(repeats)],
                            stdout=sys.stderr, stderr=subprocess.PIPE, text=True)
    sys.stderr.write(result.stderr)
    result.check_returncode()
    verify_placement(result.stderr, backend, repeats, expected_device)
    predicted = np.fromfile(target, dtype="<f4").reshape(len(values), 160)
    if not np.isfinite(predicted).all():
        raise ValueError("Non-finite GVHMR temporal output")
    return {
        "pred_x": predicted[:, :151],
        "pred_cam": predicted[:, 151:154],
        "static_conf_logits": predicted[:, 154:],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="operation", required=True)
    convert = actions.add_parser("convert")
    predict = actions.add_parser("predict")
    for action in (convert, predict):
        action.add_argument("--probe", type=Path, required=True)
        action.add_argument("--weights", type=Path, required=True)
    convert.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--conditions", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--backend", choices=PLATFORMS, default="cpu")
    args = parser.parse_args()
    args.probe = args.probe.resolve(strict=True)
    if args.operation == "convert":
        if args.weights.exists() or args.weights.suffix.lower() != ".safetensors":
            parser.error("--weights must be a new .safetensors file")
        save_weights(load_checkpoint(args.checkpoint, schema_for(args.probe)), args.weights)
        return
    if args.output.exists() or args.output.suffix.lower() != ".npz":
        parser.error("--output must be a new .npz file")
    conditions = read_conditions(args.conditions)
    with scratch_directory() as work:
        os.environ.update(run_environment(work, FORK_TOOLS))
        result = run(args.probe, args.backend, args.weights, conditions, work, "motion")
        with args.output.open("xb") as output:
            np.savez(output, **result)


if __name__ == "__main__":
    main()
