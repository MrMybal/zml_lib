"""Partition a local TRELLIS.2 or Pixal3D sparse VAE into native stages."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from safetensors import safe_open
from safetensors.torch import save_file
import torch


PROFILES = {
    "shape_encode": ("FlexiDualGridVaeEncoder", True),
    "shape_decode": ("FlexiDualGridVaeDecoder", False),
    "texture_encode": ("SparseUnetVaeEncoder", True),
    "texture_decode": ("SparseUnetVaeDecoder", False),
}
WIDTHS = [1024, 512, 256, 128, 64]
BLOCKS = [4, 16, 8, 4, 0]


def schema_for(probe, profile, level, phase):
    text = subprocess.check_output([probe, "schema", profile, str(level), phase], text=True)
    result = {}
    for line in text.splitlines():
        name, *dims = line.split("\t")
        if name in result:
            raise ValueError(f"Duplicate schema tensor {name}")
        result[name] = tuple(map(int, dims))
    return result


def check_config(path, profile):
    data = json.loads(path.read_text(encoding="utf-8"))
    cls, encoding = PROFILES[profile]
    config = data["args"]
    widths, blocks = (WIDTHS[::-1], BLOCKS[::-1]) if encoding else (WIDTHS, BLOCKS)
    resampling = ("down_block_type", "SparseResBlockS2C3d") if encoding else ("up_block_type", "SparseResBlockC2S3d")
    if (data["name"] != cls or config["model_channels"] != widths or config["latent_channels"] != 32
            or config["num_blocks"] != blocks or config["block_type"] != ["SparseConvNeXtBlock3d"] * 5
            or config["block_args"] != [{}] * 5 or config[resampling[0]] != [resampling[1]] * 4):
        raise ValueError("Unsupported TRELLIS.2 sparse VAE configuration")
    storage = "fp8" if config.get("use_fp8") is True else "fp16" if config.get("use_fp16") is True else None
    if storage is None:
        raise ValueError("Expected FP8 TRELLIS.2 or FP16 Pixal3D weights")
    if profile == "texture_decode" and (config.get("pred_subdiv") is not False or config["out_channels"] != 6):
        raise ValueError("The texture decoder must use the shape subdivisions")
    return storage


def prepare(args):
    storage = check_config(args.config, args.profile)
    target = args.output.resolve()
    if target.exists():
        raise FileExistsError(target)
    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)
    with TemporaryDirectory(prefix="vae-prepare-", dir=target.parent) as tmp:
        staging = Path(tmp) / "weights"
        staging.mkdir()
        written = set()
        with safe_open(args.checkpoint, framework="pt", device="cpu") as source:
            for level in range(5):
                for phase in (("pre", "post") if level != 4 else ("pre",)):
                    stage = {}
                    for key, shape in schema_for(args.probe, args.profile, level, phase).items():
                        if key in written:
                            raise ValueError(f"Duplicate stage weights for {key}")
                        value = source.get_tensor(key)
                        if (storage == "fp8" and value.dtype != torch.float8_e4m3fn
                                or storage == "fp16" and value.dtype not in (torch.float32, torch.float16)):
                            raise ValueError(f"Unexpected {storage} storage for {key}: {value.dtype}")
                        if value.ndim == 5 and len(shape) == 2:
                            out, d, h, w, inputs = value.shape
                            if (d, h, w) != (3, 3, 3):
                                raise ValueError(f"{key} has an unexpected sparse kernel")
                            value = value.reshape(out, 27 * inputs)
                        if tuple(value.shape) != shape or not torch.isfinite(value.float()).all().item():
                            raise ValueError(f"Expected finite {key} with shape {shape}")
                        stage[key] = value.contiguous() if value.ndim == 2 and storage == "fp8" else value.float().contiguous()
                        written.add(key)
                    save_file(stage, staging / f"{level}-{phase}.safetensors")
            if written != set(source.keys()):
                raise ValueError(f"Unused checkpoint tensors: {sorted(set(source.keys()) - written)}")

        with args.checkpoint.open("rb") as file:
            digest = sha256()
            for data in iter(lambda: file.read(1 << 20), b""):
                digest.update(data)
        (staging / "manifest.json").write_text(json.dumps({
            "profile": args.profile, "source_sha256": digest.hexdigest(), "tensors": len(written), "storage": storage,
        }, indent=2) + "\n", encoding="utf-8")
        staging.rename(target)
    print(f"{args.profile}: {len(written)} tensors across nine native stages")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "checkpoint", "probe", "output"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
