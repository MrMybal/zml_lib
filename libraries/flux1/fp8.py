"""Read ComfyUI E4M3FN scalar weight scales for the FLUX denoiser."""

import json
import math

import torch


FORMAT = "e4m3_scaled"
SCALE_KEYS = ("weight_scale", "scale_weight")
INPUT_SCALE_KEYS = ("input_scale", "scale_input")
EXTRA_KEYS = ("weight_scale_2", "pre_quant_scale", "weight_s_rel", "weight_s_channel")
QUANT_KEYS = (*SCALE_KEYS, *INPUT_SCALE_KEYS, *EXTRA_KEYS, "comfy_quant")
FLOAT_DTYPES = {"F32", "F16", "BF16"}


def layers_metadata(checkpoint):
    info = (checkpoint.metadata() or {}).get("_quantization_metadata")
    if info is None:
        return {}
    decoded = json.loads(info)
    if not isinstance(decoded, dict) or not isinstance(decoded.get("layers"), dict):
        raise ValueError("Expected ComfyUI quantization metadata with a layers object")
    return decoded["layers"]


def find_scale_key(keys, layer, suffixes):
    matched = [layer + "." + suffix for suffix in suffixes if layer + "." + suffix in keys]
    if len(matched) > 1:
        raise ValueError(f"Ambiguous scale keys for {layer}")
    return matched[0] if matched else None


def read_scale(checkpoint, name, *, weight=False):
    part = checkpoint.get_slice(name)
    if tuple(part.get_shape()) not in ((), (1,)) or part.get_dtype() not in FLOAT_DTYPES:
        raise ValueError(f"Expected a scalar floating scale for {name}")
    result = float(checkpoint.get_tensor(name).item())
    limit = torch.finfo(torch.float32).max / (448 if weight else 1)
    if not math.isfinite(result) or not 0 < result <= limit:
        raise ValueError(f"Expected a positive finite scale for {name}")
    return result


def quantization_format(checkpoint, keys, layers, prefix, layer):
    local = layer.removeprefix(prefix)
    configurations = [layers[key] for key in dict.fromkeys((layer, local)) if key in layers]
    key = layer + ".comfy_quant"
    if key in keys:
        info = checkpoint.get_slice(key)
        shape = info.get_shape()
        if info.get_dtype() != "U8" or len(shape) != 1 or not 0 < shape[0] <= 4096:
            raise ValueError(f"Invalid ComfyUI quantization metadata tensor {key}")
        configurations.append(json.loads(checkpoint.get_tensor(key).numpy().tobytes()))
    if not configurations:
        return None
    if any(not isinstance(config, dict) or config.get("format") != "float8_e4m3fn" for config in configurations):
        raise ValueError(f"Unsupported quantization for {layer}. Expected float8_e4m3fn")
    return "float8_e4m3fn"


def has_scaled_weights(checkpoint, prefix, schema):
    keys = checkpoint.keys()
    marker = prefix + "scaled_fp8"
    suffixes = tuple("." + key for key in QUANT_KEYS)
    if marker in keys or any(name.startswith(prefix) and name.endswith(suffixes) for name in keys):
        return True
    layers = layers_metadata(checkpoint)
    return any(name[:-7] in layers or prefix + name[:-7] in layers for name in schema if name.endswith(".weight"))


def plan(checkpoint, schema, prefix):
    keys = set(checkpoint.keys())
    layers = layers_metadata(checkpoint)
    marker = prefix + "scaled_fp8"
    legacy_e4m3 = False
    if marker in keys:
        dtype = checkpoint.get_slice(marker).get_dtype()
        if dtype not in ("F32", "F8_E4M3"):
            raise ValueError(f"Unsupported legacy scaled_fp8 marker type {dtype}")
        legacy_e4m3 = True

    weight_layers = {prefix + name[:-7] for name, shape in schema.items() if name.endswith(".weight") and len(shape) == 2}
    suffixes = tuple("." + key for key in QUANT_KEYS)
    for key in sorted(keys):
        if key.startswith(prefix) and key.endswith(suffixes) and key.rpartition(".")[0] not in weight_layers:
            raise ValueError(f"Quantization tensor does not belong to a declared linear weight: {key}")

    fields, sources, scales = {}, {}, []
    separate_scales = 0
    for name, shape in schema.items():
        key = prefix + name
        part = checkpoint.get_slice(key)
        dtype = part.get_dtype()
        if tuple(part.get_shape()) != shape:
            raise ValueError(f"Unexpected shape for {key}: {part.get_shape()}")
        storage = "f32"
        scale = 1.0
        if name.endswith(".weight") and len(shape) == 2:
            layer = key[:-7]
            if any(layer + "." + suffix in keys for suffix in EXTRA_KEYS):
                raise ValueError(f"Unsupported secondary or block scale for {layer}")
            scheme = quantization_format(checkpoint, keys, layers, prefix, layer)
            factor_key = find_scale_key(keys, layer, SCALE_KEYS)
            input_key = find_scale_key(keys, layer, INPUT_SCALE_KEYS)
            if (scheme or input_key) and factor_key is None:
                raise ValueError(f"Missing weight scale for {layer}")
            is_fp8 = dtype == "F8_E4M3" or (dtype == "U8" and (scheme is not None or legacy_e4m3))
            if factor_key is not None and not is_fp8:
                raise ValueError(f"Expected E4M3FN weight storage for {key}")
            if is_fp8:
                storage = "e4m3"
                if factor_key:
                    scale = read_scale(checkpoint, factor_key, weight=True)
                    separate_scales += 1
                if input_key:
                    # Calibration scales are for quantized activations. This graph uses FP32 activations.
                    read_scale(checkpoint, input_key)
        if storage == "f32" and dtype not in FLOAT_DTYPES:
            raise ValueError(f"Expected a floating ordinary tensor for {key}, found {dtype}")
        fields[name] = (shape, storage)
        sources[name] = key
        scales.append(scale)
    if separate_scales == 0:
        raise ValueError("The denoiser contains no E4M3FN weights with separate scales")
    return fields, sources, scales


def tensor_value(checkpoint, key, storage):
    value = checkpoint.get_tensor(key)
    if storage == "e4m3":
        return value.view(torch.uint8)
    value = value.float()
    if not torch.isfinite(value).all().item():
        raise ValueError(f"Nonfinite FP32 tensor {key}")
    return value
