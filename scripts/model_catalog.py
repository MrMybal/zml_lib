"""Identify supported local checkpoints and prepare them with the existing ZML converters."""

import argparse
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import is_zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libraries.realesrgan.scripts import checkpoints as rrdb

RRDB_DIR = ROOT / "libraries/realesrgan/scripts"
SD_DIR = ROOT / "libraries/stable_diffusion"
SD_PROFILES = {"sd15": 4, "sd15-inpaint": 9}
COMPONENTS = {"clip_l": ("clip", "clip_text_probe", "f32"), "t5_xxl": ("t5", "t5_encoder", "f16"),
              "flux1_vae": ("flux1", "flux1_probe", "f32")}
PROFILES = {**dict.fromkeys(rrdb.RUNTIME_VARIANTS, "realesrgan"), **dict.fromkeys(SD_PROFILES, "stable_diffusion"),
            **{profile: library for profile, (library, _, _) in COMPONENTS.items()}}
LEGACY_RRDB = "4x_NMKD-Siax_200k"
FORMATS = {".safetensors", ".pth", ".ckpt"}
SCAN_FORMATS = FORMATS | {".pt", ".sft", ".gguf", ".onnx", ".part"}
FLOATING = {"F32", "F16", "BF16"}
INPUT_ERRORS = (OSError, ValueError, RuntimeError, pickle.UnpicklingError)
CACHE_VERSION = 1


@dataclass(frozen=True)
class Match:
    profile: str
    layout: str
    ready: bool = False

    @property
    def library(self):
        return PROFILES[self.profile]

    @property
    def precision(self):
        return COMPONENTS[self.profile][2] if self.profile in COMPONENTS else "f32"

    def report(self, source):
        return {"path": str(source), "profile": self.profile, "library": self.library,
                "layout": self.layout, "precision": self.precision,
                "status": "ready" if self.ready else "needs_preparation"}


def tensor_header(path):
    from safetensors import safe_open, SafetensorError

    try:
        with safe_open(path, framework="numpy") as handle:
            header = {}
            for name in handle.keys():
                view = handle.get_slice(name)
                header[name] = tuple(view.get_shape()), view.get_dtype()
            return header
    except SafetensorError as exc:
        raise ValueError(f"Invalid SafeTensors file: {path.name}") from exc


def torch_header(path):
    import torch

    raw = torch.load(path, map_location="meta", weights_only=True, mmap=is_zipfile(path))
    state, section = rrdb.select_state(raw, allow_flat=True)
    if section == "state_dict" and "state_dict" in state:
        state = state["state_dict"]
        if not isinstance(state, Mapping):
            raise ValueError("Expected a state_dict of checkpoint tensors")
        section = "nested_state_dict"
    types = {torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16"}
    return {name: (tuple(value.shape), types.get(value.dtype, "FLOAT" if value.is_floating_point() else "OTHER"))
            for name, value in state.items() if isinstance(name, str)
            and isinstance(value, torch.Tensor) and value.layout == torch.strided}, section, len(state)


def matches(shapes, header, allowed):
    return all(name in header and header[name][0] == shape and header[name][1] in allowed
               for name, shape in shapes.items())


def rrdb_profile(header, section, count):
    if section == "nested_state_dict" or len(header) != count:
        return None
    layout = "rrdbnet"
    if "model.0.weight" in header:
        try:
            header = rrdb.normalize_legacy(header, rrdb.MODELS[LEGACY_RRDB]["blocks"])
        except ValueError:
            return None
        candidates = ("RealESRGAN_x4plus",)
        layout = "esrgan_sequential"
    else:
        candidates = rrdb.RUNTIME_VARIANTS
    for profile in candidates:
        expected = rrdb.expected_shapes(profile)
        if expected.keys() == header.keys() and matches(expected, header, FLOATING | {"FLOAT"}):
            ready = layout == "rrdbnet" and section == "safetensors" and all(info[1] == "F32" for info in header.values())
            return Match(profile, layout, ready)
    return None


def sd_signatures(channels):
    return {
        "unet": (686, {"input_blocks.0.0.weight": (320, channels, 3, 3), "out.2.weight": (4, 320, 3, 3),
                       "input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight": (320, 768)}),
        "clip": (196, {"text_model.embeddings.token_embedding.weight": (49408, 768),
                       "text_model.embeddings.position_embedding.weight": (77, 768)}),
        "vae": (140, {"decoder.conv_in.weight": (512, 4, 3, 3), "post_quant_conv.weight": (4, 4, 1, 1)}),
        "encoder": (108, {"encoder.conv_in.weight": (128, 3, 3, 3), "quant_conv.weight": (8, 8, 1, 1)}),
    }


def sd_profile(header):
    parts = {name: {} for name in sd_signatures(4)}
    prefixes = {"model.diffusion_model.": "unet", "cond_stage_model.transformer.": "clip", "first_stage_model.": "vae"}
    for name, desc in header.items():
        for prefix, role in prefixes.items():
            if name.startswith(prefix):
                key = name.removeprefix(prefix)
                if role == "clip" and key == "text_model.embeddings.position_ids":
                    break
                if role == "vae" and key.startswith(("encoder.", "quant_conv.")):
                    parts["encoder"][key] = desc
                else:
                    parts[role][key] = desc
                break
    for profile, channels in SD_PROFILES.items():
        if all(len(parts[role]) == count and matches(shapes, parts[role], FLOATING)
               and all(dtype in FLOATING for _, dtype in parts[role].values())
               for role, (count, shapes) in sd_signatures(channels).items()):
            return Match(profile, "compvis")
    return None


def component_profile(header):
    clip = {key: desc for key, desc in header.items() if key != "text_model.embeddings.position_ids"}
    count, anchors = sd_signatures(4)["clip"]
    if len(clip) == count and matches(anchors, clip, FLOATING) and all(dtype in FLOATING for _, dtype in clip.values()):
        return Match("clip_l", "transformers_clip",
                     len(header) == count and all(dtype == "F32" for _, dtype in clip.values()))

    token_table = header.get("shared.weight") or header.get("encoder.embed_tokens.weight")
    t5 = {name: desc for name, desc in header.items() if name != "encoder.embed_tokens.weight"}
    if "shared.weight" not in t5 and token_table:
        t5["shared.weight"] = token_table
    anchors = {"shared.weight": (32128, 4096), "encoder.final_layer_norm.weight": (4096,),
               "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight": (32, 64),
               "encoder.block.23.layer.1.DenseReluDense.wi_0.weight": (10240, 4096)}
    if (len(t5) == 219 and matches(anchors, t5, FLOATING)
            and all(dtype in FLOATING for _, dtype in header.values())
            and all(name == "shared.weight" or name.startswith("encoder.") for name in header)
            and ("encoder.embed_tokens.weight" not in header or header["encoder.embed_tokens.weight"][0] == (32128, 4096))):
        return Match("t5_xxl", "transformers_t5",
                     len(header) == 219 and "shared.weight" in header and all(dtype == "F16" for _, dtype in t5.values()))

    decoder = {name: desc for name, desc in header.items() if name.startswith("decoder.")}
    anchors = {"decoder.conv_in.weight": (512, 16, 3, 3), "decoder.conv_out.weight": (3, 128, 3, 3),
               "decoder.mid.attn_1.q.weight": (512, 512, 1, 1)}
    if (len(decoder) == 138 and matches(anchors, decoder, FLOATING)
            and all(dtype in FLOATING for _, dtype in decoder.values())
            and all(name.startswith(("encoder.", "decoder.")) for name in header)):
        return Match("flux1_vae", "flux1_autoencoder",
                     len(header) == 138 and all(dtype == "F32" for _, dtype in decoder.values()))
    return None


def bundle_profile(path):
    manifest = json.loads((path / "sd15.json").read_text(encoding="utf-8"))
    profile = manifest.get("profile") if isinstance(manifest, dict) else None
    if not isinstance(profile, str) or profile not in SD_PROFILES or manifest != {"profile": profile, "storage": "f32"}:
        raise ValueError("Expected a prepared SD 1.5 FP32 bundle")
    for role, (count, shapes) in sd_signatures(SD_PROFILES[profile]).items():
        if role == "encoder" and profile == "sd15":
            continue
        name = role + ".safetensors"
        header = tensor_header(path / name)
        if len(header) != count or not matches(shapes, header, {"F32"}) or any(dtype != "F32" for _, dtype in header.values()):
            raise ValueError(f"Invalid prepared SD 1.5 component: {name}")
    return Match(profile, "sd15_bundle", ready=True)


def read_signature(path):
    if path.suffix.lower() == ".safetensors":
        header = tensor_header(path)
        return header, "safetensors", len(header)
    return torch_header(path)


def detect(header, section, count, suffix):
    if section == "safetensors":
        component = component_profile(header)
        if component:
            return component
    match = rrdb_profile(header, section, count)
    if match and (section in ("params_ema", "params", "safetensors") or match.layout == "esrgan_sequential"):
        return match
    if suffix in (".ckpt", ".safetensors") and section in ("safetensors", "state_dict", "nested_state_dict"):
        match = sd_profile(header)
        if match:
            return match
    return None


def identify(path):
    path = Path(path)
    if path.is_dir():
        return bundle_profile(path)
    if not path.is_file():
        raise ValueError("Checkpoint must be a regular local file")
    suffix = path.suffix.lower()
    if suffix not in FORMATS:
        raise ValueError("Expected a local .safetensors, .pth, .ckpt or prepared SD 1.5 bundle")
    match = detect(*read_signature(path), suffix)
    if match is None:
        raise ValueError("No supported checkpoint profile matches the tensor signature")
    return match


def scan(root):
    if not root.is_dir():
        raise ValueError(f"Expected a model directory: {root}")
    entries = []
    for folder, subdirs, names in os.walk(root, followlinks=False):
        directory = Path(folder)
        # Skip junctions and links in the selected directory.
        subdirs[:] = sorted(name for name in subdirs if (directory / name).resolve() == directory / name)
        for name in sorted(names):
            path = directory / name
            suffix = path.suffix.lower()
            if suffix not in SCAN_FORMATS or not path.is_file() or path.resolve() != path:
                continue
            relative = path.relative_to(root).as_posix()
            row = {"path": relative, "status": "unsupported"}
            try:
                row["bytes"] = path.stat().st_size
                if suffix == ".part":
                    row["status"] = "partial_download"
                elif suffix in FORMATS:
                    header, section, count = read_signature(path)
                    row.update(tensors=count, dtypes=dict(sorted(Counter(dtype for _, dtype in header.values()).items())))
                    match = detect(header, section, count, suffix)
                    if match:
                        row.update(match.report(relative))
            except INPUT_ERRORS as exc:
                row.update(status="unreadable", error=str(exc))
            entries.append(row)
    return {"root": str(root), "counts": dict(sorted(Counter(row["status"] for row in entries).items())),
            "entries": sorted(entries, key=lambda row: row["path"].casefold())}


def conversion(match, source, bin_dir):
    if match.library == "realesrgan":
        model = LEGACY_RRDB if match.layout == "esrgan_sequential" else match.profile
        converter = RRDB_DIR / "convert.py"
        return ([converter, "--architecture-only", "--model", model, "--checkpoint", source],
                [converter, RRDB_DIR / "checkpoints.py", RRDB_DIR.parent / "models/provenance.json"])
    if match.profile in COMPONENTS:
        library, executable, precision = COMPONENTS[match.profile]
        probe = require_probe(bin_dir, library, executable)
        if library == "t5":
            converter = ROOT / "libraries/t5/prepare.py"
            command = [converter, "--profile", match.profile, "--probe", probe, "--source", source, "--storage", precision]
            return command, [converter, probe]
        converter = ROOT / "scripts/prepare_image_component.py"
        command = [converter, "--profile", match.profile, "--probe", probe, "--source", source]
        return command, [converter, SD_DIR / "prepare.py", probe]
    probe = require_probe(bin_dir, "stable_diffusion", "sd15_probe")
    converter = SD_DIR / "prepare.py"
    return ([converter, source, "--profile", match.profile, "--bin-dir", bin_dir], [converter, probe])


def require_probe(bin_dir, library, executable):
    relative = Path("libraries") / library / (executable + ".exe")
    if bin_dir is None:
        raise ValueError(f"Preparation requires --bin-dir containing {relative}")
    probe = bin_dir / relative
    if not probe.is_file():
        raise ValueError(f"Probe not found: {probe}")
    return probe


def file_state(path):
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def fingerprint_artifacts(directory, names):
    artifacts = {}
    for name in names:
        path = directory / name
        if not path.is_file() or path.resolve() != path or path.stat().st_size == 0:
            raise ValueError(f"Missing or invalid prepared artifact: {name}")
        artifacts[name] = {"bytes": path.stat().st_size, "sha256": rrdb.sha256(path)}
    return artifacts


def check_cache(target, record):
    try:
        manifest = target / "catalog.json"
        if target.resolve() != target or manifest.resolve() != manifest:
            raise ValueError("Redirected cache entry")
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or any(metadata.get(key) != value for key, value in record.items()):
            raise ValueError("Cache manifest does not match")
        if metadata["files"] != fingerprint_artifacts(target, record["artifacts"]):
            raise ValueError("Cached file fingerprints do not match")
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError(f"Invalid cache entry: {target}. Choose another cache directory or remove this entry.") from exc


def prepare(source, cache_dir, bin_dir=None):
    source = Path(source).resolve(strict=True)
    cache_dir = Path(cache_dir).resolve()
    if bin_dir is not None:
        bin_dir = Path(bin_dir).resolve()
    start = file_state(source)
    match = identify(source)
    report = match.report(source)
    if match.ready:
        return report
    args, version_files = conversion(match, source, bin_dir)
    record = {"schema": CACHE_VERSION, "profile": match.profile, "layout": match.layout, "precision": match.precision,
              "source_sha256": rrdb.sha256(source), "converter_sha256": [rrdb.sha256(path) for path in version_files]}
    if file_state(source) != start:
        raise ValueError("Checkpoint changed during inspection")
    directory_output = match.library == "stable_diffusion"
    record["artifacts"] = (["model.safetensors"] if not directory_output else
                           ["bundle/sd15.json", "bundle/unet.safetensors", "bundle/clip.safetensors", "bundle/vae.safetensors"])
    if match.profile == "sd15-inpaint":
        record["artifacts"].append("bundle/encoder.safetensors")
    key = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{match.profile}-{key}"
    output = "bundle" if directory_output else "model.safetensors"
    result = {**report, "source_sha256": record["source_sha256"], "path": str(target / output)}
    if target.exists() or target.is_symlink():
        check_cache(target, record)
        return {**result, "status": "cached"}
    with TemporaryDirectory(prefix="prepare-", dir=cache_dir) as temp:
        staged = Path(temp) / "entry"
        staged.mkdir()
        subprocess.run([sys.executable, "-B", *args, "--output", staged / output], check=True, stdout=sys.stderr)
        if file_state(source) != start:
            raise ValueError("Checkpoint changed during preparation")
        metadata = {**record, "files": fingerprint_artifacts(staged, record["artifacts"])}
        (staged / "catalog.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        try:
            staged.rename(target)
        except OSError:
            if not (target.exists() or target.is_symlink()):
                raise
            check_cache(target, record)
            return {**result, "status": "cached"}
    return {**result, "status": "prepared"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List supported runtime profiles")
    commands.add_parser("scan", help="Inventory local model directories").add_argument("directories", type=Path, nargs="+")
    commands.add_parser("identify", help="Inspect a local checkpoint").add_argument("checkpoint", type=Path)
    convert = commands.add_parser("prepare", help="Prepare or reuse a checked cache entry")
    convert.add_argument("checkpoint", type=Path)
    convert.add_argument("--cache-dir", required=True, type=Path)
    convert.add_argument("--bin-dir", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "list":
            result = [{"profile": profile, "library": library} for profile, library in PROFILES.items()]
        elif args.command == "scan":
            result = {"directories": [scan(path.resolve(strict=True)) for path in args.directories]}
        else:
            source = args.checkpoint.resolve(strict=True)
            if args.command == "identify":
                result = identify(source).report(source)
            else:
                result = prepare(source, args.cache_dir, args.bin_dir)
    except (*INPUT_ERRORS, subprocess.CalledProcessError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=True), file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
