"""Check an original LocalMesh flow checkpoint and prepare a ZML weight file."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import subprocess

from safetensors import safe_open
from safetensors.torch import save_file
import torch


PROFILES = {
    "structure": ("SparseStructureFlowModel", 8, 8, 0),
    "shape": ("SLatFlowModel", 32, 32, 0),
    "texture": ("SLatFlowModel", 64, 32, 0),
    "structure_multiview": ("SparseStructureFlowModel", 8, 8, 1024),
    "shape_multiview": ("SLatFlowModel", 32, 32, 2048),
    "texture_multiview": ("SLatFlowModel", 64, 32, 2048),
}


def dimensions(probe, profile):
    lines = subprocess.check_output([probe, profile, "schema"], text=True).splitlines()
    result = dict((name, tuple(map(int, dims))) for name, *dims in (line.split("\t") for line in lines))
    if len(result) != len(lines) or len(result) != (700 if PROFILES[profile][-1] else 640):
        raise ValueError("Unexpected flow schema size")
    return result


def validate_config(path, profile):
    data = json.loads(path.read_text(encoding="utf-8"))
    kind, channels, out, projected = PROFILES[profile]
    names = {kind}
    if kind == "SLatFlowModel":
        names.add("ElasticSLatFlowModel")
    if data["name"] not in names:
        raise ValueError(f"Expected {kind}")
    args = data["args"]
    expected = dict(in_channels=channels, out_channels=out, model_channels=1536, cond_channels=1024,
                    num_blocks=30, num_heads=12, pe_mode="rope", share_mod=True, qk_rms_norm=True,
                    qk_rms_norm_cross=True)
    for k, v in expected.items():
        if args.get(k) != v:
            raise ValueError(f"Expected {k}={v}, got {args.get(k)}")
    if int(1536 * args["mlp_ratio"]) != 8192:
        raise ValueError("Unexpected feed-forward width")
    if args.get("image_attn_mode", "cross") != ("proj" if projected else "cross"):
        raise ValueError("Unexpected attention mode")
    if projected and args.get("proj_in_channels", 1024) != projected:
        raise ValueError("Unexpected projected conditioning size")


def prepare(args):
    validate_config(args.config, args.profile)
    schema = dimensions(args.probe, args.profile)
    out = {}
    with safe_open(args.checkpoint, framework="pt") as source:
        extras = set(source.keys()) - schema.keys()
        if extras - {"rope_phases"} or schema.keys() - source.keys():
            raise ValueError(f"Unexpected or missing tensors: {sorted(set(source.keys()) ^ schema.keys())}")
        if "rope_phases" in extras:
            if args.profile not in ("structure", "structure_multiview"):
                raise ValueError("Only the dense volume has a cached rotary table")
            phases = source.get_tensor("rope_phases")
            coords = torch.cartesian_prod(*(torch.arange(16, dtype=torch.float32),) * 3)
            frequencies = 1 / 10000 ** (torch.arange(21, dtype=torch.float32) / 21)
            angles = (coords[:, :, None] * frequencies).flatten(1)
            expected = torch.polar(torch.ones((4096, 64)), torch.nn.functional.pad(angles, (0, 1)))
            torch.testing.assert_close(phases, expected)
        for name, shape in schema.items():
            saved = source.get_slice(name)
            kind = saved.get_dtype()
            if tuple(saved.get_shape()) != shape or kind not in ("F32", "F16", "BF16", "F8_E4M3"):
                raise ValueError(f"Unsupported dtype or shape for {name}")
            value = source.get_tensor(name)
            if kind == "F8_E4M3" and name.endswith(".weight") and len(shape) == 2:
                # Keep the matrix FP8. ZML decodes the exact E4M3FN value.
                out[name] = value
            else:
                out[name] = value.float()
    with args.checkpoint.open("rb") as stream:
        original = file_digest(stream, "sha256").hexdigest()
    save_file(out, args.output, metadata={"profile": args.profile, "source_sha256": original})
    print(json.dumps({"profile": args.profile, "tensors": len(out), "source_sha256": original}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("probe", "config", "checkpoint", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    args = parser.parse_args()
    args.probe = args.probe.resolve(strict=True)
    if args.output.exists() or args.output.suffix != ".safetensors":
        parser.error("--output must be a new SafeTensors file")
    prepare(args)


if __name__ == "__main__":
    main()
