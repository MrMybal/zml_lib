"""Decode ZML SLAT features with a local TRELLIS v1 representation head."""

import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
from safetensors.torch import load_file

from flow_io import FORK_TOOLS, finite_fp32, topology
from build import run_environment, scratch_directory


FORMATS = {
    "mesh": ("SLatMeshDecoder", ".npz"),
    "gaussian": ("SLatGaussianDecoder", ".ply"),
    "radiance": ("SLatRadianceFieldDecoder", ".npz"),
}


def load_slat(path):
    archive = np.load(path, allow_pickle=False)
    if isinstance(archive, np.ndarray):
        raise ValueError("Expected a SLAT .npz archive")
    with archive:
        coords = archive["coordinates"]
        topology(coords)
        return coords.astype(np.int32), finite_fp32(archive["features"], (len(coords), 8), "SLAT")


def write_representation(representation, kind, target):
    def cpu(tensor):
        return tensor.detach().cpu().numpy()

    if kind == "gaussian":
        representation.save_ply(str(target))
    elif kind == "mesh":
        if not representation.success:
            raise ValueError("TRELLIS returned an empty mesh")
        result = {"vertices": cpu(representation.vertices), "faces": cpu(representation.faces)}
        if representation.vertex_attrs is not None:
            result["vertex_attrs"] = cpu(representation.vertex_attrs)
        np.savez(target, **result)
    else:
        values = {key: cpu(getattr(representation, key))
                  for key in ("position", "depth", "trivec", "density", "features_dc")}
        np.savez(target, **values, aabb=np.asarray([-0.5, -0.5, -0.5, 1, 1, 1], dtype=np.float32))


@torch.inference_mode()
def decode(args, work, model_type, suffix):
    os.environ.update(run_environment(work, FORK_TOOLS))
    os.environ.update({
        "CUDA_CACHE_PATH": str(work / "cuda-cache"),
        "TORCH_EXTENSIONS_DIR": str(work / "torch-extensions"),
        "TRITON_CACHE_DIR": str(work / "triton"),
        "HF_HUB_OFFLINE": "1",
    })
    config_path = args.checkpoint.with_suffix(".json")
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get("name") != model_type or not isinstance(config.get("args"), dict):
        raise ValueError(f"Expected the local TRELLIS v1 {model_type} configuration")
    coords, features = load_slat(args.slat)
    if not torch.cuda.is_available():
        raise RuntimeError("The TRELLIS v1 representation runtime requires CUDA")
    sys.path.insert(0, str(args.upstream))
    from trellis import models
    from trellis.modules import sparse

    with torch.cuda.device(args.gpu):
        model = getattr(models, model_type)(**config["args"])
        model.load_state_dict(load_file(args.checkpoint), strict=True)
        model.cuda().eval()
        latent = sparse.SparseTensor(feats=torch.from_numpy(features).cuda(), coords=torch.from_numpy(coords).cuda())
        representation, = model(latent)
        completed = work / ("representation" + suffix)
        write_representation(representation, args.format, completed)
    with completed.open("rb") as source, args.output.open("xb") as output:
        shutil.copyfileobj(source, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("slat", "upstream", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--format", choices=tuple(FORMATS), required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    args.upstream = args.upstream.resolve(strict=True)
    args.checkpoint = args.checkpoint.resolve(strict=True)
    args.output = args.output.resolve()
    model, suffix = FORMATS[args.format]
    if not (args.upstream / "trellis" / "models" / "__init__.py").is_file():
        parser.error("--upstream must be a local TRELLIS v1 checkout")
    if args.checkpoint.suffix.lower() != ".safetensors":
        parser.error("--checkpoint must be a local .safetensors file with its matching JSON")
    if args.output.exists() or args.output.suffix.lower() != suffix:
        parser.error(f"--output must be a new {suffix} file for {args.format}")
    with redirect_stdout(sys.stderr), scratch_directory() as work:
        decode(args, work, model, suffix)
    print(json.dumps({"output": str(args.output), "format": args.format}))


if __name__ == "__main__":
    main()
