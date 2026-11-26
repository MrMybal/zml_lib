"""Extract ZML weights for FLUX.1 Schnell or Canny Dev from local checkpoints."""

import argparse
import json
from math import prod
from pathlib import Path
import struct
import subprocess
from tempfile import TemporaryDirectory

import torch
from safetensors import safe_open
from safetensors.torch import save_file

if __package__:
    from . import fp8
else:
    import fp8


STORAGE = {"f32": ("F32", 4, torch.float32), "f16": ("F16", 2, torch.float16),
           "e4m3": ("F8_E4M3", 1, torch.float8_e4m3fn)}
INPUT_TYPES = {"F32", "F16", "BF16", "F8_E4M3"}
PROFILES = ("schnell", "canny_dev")


def probe_path(bin_dir, folder, name):
    return (bin_dir / "libraries" / folder / (name + ".exe")).resolve(strict=True)


def schema_for(probe, *profile):
    text = subprocess.check_output([probe, *profile, "schema"], text=True, encoding="utf-8")
    return {name: tuple(map(int, dims)) for name, *dims in (row.split("\t") for row in text.splitlines())}


def selected_tensors(checkpoint, schema, prefix):
    names = set(checkpoint.keys())
    selected = {}
    for name, shape in schema.items():
        source = prefix + name
        part = checkpoint.get_slice(source)
        if tuple(part.get_shape()) != shape or part.get_dtype() not in INPUT_TYPES:
            raise ValueError(f"Unexpected shape or precision for {source}: {part.get_shape()}, {part.get_dtype()}")
        if source.endswith(".weight") and any(
            source.removesuffix(".weight") + suffix in names for suffix in (".weight_scale", ".scale_weight")
        ):
            raise ValueError(f"Scaled FP8 tensors require a separate conversion: {source}")
        selected[name] = source
    return selected


def write_tensors(output, storage, fields, values):
    """Stream tensors in header order, including raw bytes for E4M3FN."""
    header = {"__metadata__": {"format": "pt", "storage": storage}}
    offset = 0
    for name, (shape, precision) in fields.items():
        label, item_size, _ = STORAGE[precision]
        end = offset + item_size * prod(shape)
        header[name] = {"dtype": label, "shape": shape, "data_offsets": [offset, end]}
        offset = end
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    with output.open("xb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        for _, value in zip(fields, values, strict=True):
            value.contiguous().numpy().tofile(stream)


def write_checkpoint(checkpoint, schema, prefix, output, storage):
    selected = selected_tensors(checkpoint, schema, prefix)
    label, _, dtype = STORAGE[storage]

    def values():
        for name, source in selected.items():
            value = checkpoint.get_tensor(source)
            if storage == "e4m3":
                if value.dtype != dtype:
                    raise ValueError(f"Expected source E4M3FN data for {name}")
                value = value.view(torch.uint8)
            else:
                value = value.to(dtype)
                if not torch.isfinite(value).all().item():
                    raise ValueError(f"Nonfinite {label} data in {name}")
            yield value

    write_tensors(output, storage, {name: (shape, storage) for name, shape in schema.items()}, values())


def quantize_denoiser(checkpoint, schema, prefix, output):
    selected = selected_tensors(checkpoint, schema, prefix)
    if any(checkpoint.get_slice(source).get_dtype() not in fp8.FLOAT_DTYPES for source in selected.values()):
        raise ValueError("Expected unquantized floating Canny Dev weights for the scaled FP8 conversion")
    fields = {name: (shape, "e4m3" if len(shape) == 2 else "f32") for name, shape in schema.items()}
    fields["weight_scales"] = ((len(schema),), "f32")
    scales = []

    def values():
        for name, source in selected.items():
            value = fp8.tensor_value(checkpoint, source, "f32")
            scale = 1.0
            if fields[name][1] == "e4m3":
                maximum = value.abs().amax().item()
                scale = max(maximum / 448.0, torch.finfo(torch.float32).tiny) if maximum else 1.0
                value = (value / scale).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
            scales.append(scale)
            yield value
        yield torch.tensor(scales, dtype=torch.float32)

    write_tensors(output, fp8.FORMAT, fields, values())


def write_denoiser(checkpoint, schema, prefix, output, storage, *, profile="schnell"):
    has_scales = fp8.has_scaled_weights(checkpoint, prefix, schema)
    if storage == "auto":
        storage = fp8.FORMAT if has_scales or profile == "canny_dev" else "e4m3"
    if storage != fp8.FORMAT:
        if has_scales:
            raise ValueError("Use --storage auto or e4m3_scaled for a denoiser with separate FP8 scales")
        write_checkpoint(checkpoint, schema, prefix, output, storage)
        return storage
    if profile == "canny_dev" and not has_scales:
        quantize_denoiser(checkpoint, schema, prefix, output)
        return storage
    fields, sources, scales = fp8.plan(checkpoint, schema, prefix)
    fields["weight_scales"] = ((len(schema),), "f32")

    def values():
        for name, source in sources.items():
            yield fp8.tensor_value(checkpoint, source, fields[name][1])
        yield torch.tensor(scales, dtype=torch.float32)

    write_tensors(output, storage, fields, values())
    return storage


def denoiser_prefix(checkpoint, profile="schnell"):
    keys = set(checkpoint.keys())
    prefixes = [prefix for prefix in ("model.diffusion_model.", "diffusion_model.", "")
                if prefix + "img_in.weight" in keys]
    if len(prefixes) != 1:
        raise ValueError("Expected one FLUX.1 denoiser in BFL/ComfyUI tensor layout")
    prefix, = prefixes
    if any(name.startswith(prefix + "guidance_in.") for name in keys) != (profile == "canny_dev"):
        raise ValueError(f"Unexpected guidance weights for {profile}")
    return prefix


def prepare(source, taef1, output, bin_dir, storage, denoiser=None, *, profile="schnell", components=None):
    flux = probe_path(bin_dir, "flux1", "flux1_probe")
    t5 = probe_path(bin_dir, "t5", "t5_encoder")
    schema = schema_for(flux, profile, "e4m3")
    if profile == "canny_dev":
        if source is not None or taef1 is not None or denoiser is None or components is None:
            raise ValueError("Canny Dev requires separate denoiser, CLIP-L, T5-XXL and FLUX.1 VAE checkpoints")
        clip_schema = schema_for(probe_path(bin_dir, "clip", "clip_text_probe"))
    else:
        from transformers import CLIPTextConfig, CLIPTextModel

        if source is None or components is not None:
            raise ValueError("Schnell requires its combined --source checkpoint")
        with torch.device("meta"):
            clip = CLIPTextModel(CLIPTextConfig(hidden_size=768, intermediate_size=3072, num_attention_heads=12))
        clip_schema = {name: tuple(tensor.shape) for name, tensor in clip.named_parameters()}
    if output.exists() or not output.parent.is_dir():
        raise ValueError("Choose a new output directory with an existing parent")
    with TemporaryDirectory(prefix="flux-weights-", dir=output.parent) as temp:
        bundle = Path(temp) / "bundle"
        bundle.mkdir()
        with safe_open(denoiser or source, framework="pt", device="cpu") as diffusion:
            prefix = denoiser_prefix(diffusion, profile)
            storage = write_denoiser(diffusion, schema, prefix, bundle / "flux.safetensors", storage, profile=profile)
        if profile == "canny_dev":
            clip_source, t5_source, vae_source = components
            with safe_open(clip_source, framework="pt", device="cpu") as clip:
                write_checkpoint(clip, clip_schema, "", bundle / "clip.safetensors", "f32")
            with safe_open(t5_source, framework="pt", device="cpu") as text:
                write_checkpoint(text, schema_for(t5, "t5_xxl"), "", bundle / "t5.safetensors", "f16")
            with safe_open(vae_source, framework="pt", device="cpu") as vae:
                write_checkpoint(vae, schema_for(flux, "vae"), "", bundle / "vae.safetensors", "f32")
                write_checkpoint(vae, schema_for(flux, "vae_encode"), "", bundle / "vae_encoder.safetensors", "f32")
        else:
            with safe_open(source, framework="pt", device="cpu") as weights:
                write_checkpoint(weights, schema_for(t5, "t5_xxl"), "text_encoders.t5xxl.transformer.",
                                 bundle / "t5.safetensors", "f16")
                write_checkpoint(weights, clip_schema, "text_encoders.clip_l.transformer.",
                                 bundle / "clip.safetensors", "f32")
                write_checkpoint(weights, schema_for(flux, "vae"), "vae.", bundle / "vae.safetensors", "f32")
        if taef1 is not None:
            tae_state = torch.load(taef1, map_location="cpu", weights_only=True)
            save_file({key: value.float().contiguous() for key, value in tae_state.items()}, bundle / "taef1.safetensors")
        (bundle / "flux.json").write_text(json.dumps({"profile": profile, "storage": storage, "t5_storage": "f16"}),
                                         encoding="utf-8")
        bundle.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="schnell")
    parser.add_argument("--source", type=Path, help="Combined Schnell checkpoint providing CLIP-L, T5 and VAE.")
    parser.add_argument("--denoiser", type=Path, help="Separate FLUX.1 denoiser in BFL/ComfyUI layout.")
    for name in ("clip", "t5", "vae"):
        parser.add_argument("--" + name, type=Path, help="Standalone Canny Dev component checkpoint")
    parser.add_argument("--taef1", type=Path, help="optional local TAEF1 preview decoder")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--storage", choices=("auto", *STORAGE, fp8.FORMAT), default="auto")
    args = parser.parse_args()
    components = (args.clip, args.t5, args.vae)
    if any(components) and not all(components):
        parser.error("Provide --clip, --t5 and --vae together")
    torch.set_num_threads(4)
    prepare(args.source, args.taef1, args.output, args.bin_dir, args.storage, args.denoiser,
            profile=args.profile, components=components if all(components) else None)


if __name__ == "__main__":
    main()
