"""Names, shapes and checkpoint selection for the supported RRDBNet weights."""

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re

PROVENANCE = json.loads(
    (Path(__file__).resolve().parents[1] / "models" / "provenance.json").read_text(encoding="utf-8")
)
MODELS = PROVENANCE["models"]
RUNTIME_VARIANTS = {"RealESRGAN_x4plus": 1, "RealESRGAN_x4plus_anime_6B": 2, "RealESRGAN_x2plus": 3}
LEGACY_BLOCK = re.compile(r"model\.1\.sub\.(\d+)\.RDB([1-3])\.conv([1-5])(?:\.0)?\.(weight|bias)")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def expected_shapes(variant):
    shapes = {}

    def conv(name, outputs, inputs):
        shapes[f"{name}.weight"] = (outputs, inputs, 3, 3)
        shapes[f"{name}.bias"] = (outputs,)

    conv("conv_first", 64, 12 if MODELS[variant].get("scale") == 2 else 3)
    for block in range(MODELS[variant]["blocks"]):
        for dense in range(1, 4):
            for layer in range(1, 6):
                conv(f"body.{block}.rdb{dense}.conv{layer}", 64 if layer == 5 else 32, 64 + 32 * (layer - 1))
    for name in ("conv_body", "conv_up1", "conv_up2", "conv_hr"):
        conv(name, 64, 64)
    conv("conv_last", 3, 64)
    if len(shapes) != MODELS[variant]["tensor_count"]:
        raise ValueError(f"Inconsistent tensor count for {variant}")
    return shapes


def select_state(checkpoint, *, allow_flat=False):
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must contain a mapping")
    for key in ("params_ema", "params"):
        if key in checkpoint:
            state = checkpoint[key]
            break
    else:
        if not allow_flat:
            raise ValueError("Checkpoint has no params_ema or params tensor mapping")
        key, state = "state_dict", checkpoint
    if not isinstance(state, Mapping):
        raise ValueError(f"Checkpoint has no {key} tensor mapping")
    return state, key


def checkpoint_state(path, variant):
    if path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        return load_file(path, device="cpu"), "state_dict"
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return select_state(checkpoint, allow_flat=MODELS[variant].get("layout") == "esrgan_sequential")


def normalize_legacy(state, blocks):
    layers = {
        "model.0": "conv_first",
        f"model.1.sub.{blocks}": "conv_body",
        "model.3": "conv_up1",
        "model.6": "conv_up2",
        "model.8": "conv_hr",
        "model.10": "conv_last",
    }
    converted = {}
    for name, value in state.items():
        match = LEGACY_BLOCK.fullmatch(name) if isinstance(name, str) else None
        if match:
            block, dense, layer, suffix = match.groups()
            canonical = f"body.{block}.rdb{dense}.conv{layer}.{suffix}"
        else:
            prefix, _, suffix = str(name).rpartition(".")
            if prefix not in layers or suffix not in ("weight", "bias"):
                raise ValueError(f"Unsupported ESRGAN tensor: {name}")
            canonical = f"{layers[prefix]}.{suffix}"
        if canonical in converted:
            raise ValueError(f"Duplicate ESRGAN tensor after remapping: {canonical}")
        converted[canonical] = value
    return converted


def read_checkpoint(path, variant):
    import torch

    state, key = checkpoint_state(path, variant)
    if MODELS[variant].get("layout") == "esrgan_sequential":
        state = normalize_legacy(state, MODELS[variant]["blocks"])
    shapes = expected_shapes(variant)
    missing, unexpected = shapes.keys() - state.keys(), state.keys() - shapes.keys()
    if missing or unexpected:
        raise ValueError(
            f"Tensor names for {variant}: missing={sorted(map(str, missing))}, "
            f"unexpected={sorted(map(str, unexpected))}"
        )
    tensors = {}
    for name, shape in shapes.items():
        value = state[name]
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided or not value.is_floating_point():
            raise ValueError(f"{name}: expected a dense floating-point tensor")
        if tuple(value.shape) != shape:
            raise ValueError(f"{name}: expected {shape}, got {tuple(value.shape)}")
        tensors[name] = value.detach().to(dtype=torch.float32).contiguous()
    return tensors, key
