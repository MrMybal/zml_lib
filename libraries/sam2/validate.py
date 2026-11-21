"""Convert SAM 2.0 Hiera base+ and compare prompted image masks to ComfyUI."""

import argparse
import importlib
from pathlib import Path
import sys
import tempfile
import types

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from segmentation_validation import arguments, check_checkpoint, compare, probe, raster_iou, read_rgb, sha256, write_report


EMBED = (256 * 64 * 64, 32 * 256 * 256, 64 * 128 * 128)
PARTS = ("encoder", "decoder", "memory", "attention")


def spatial_position(channels):
    import torch

    y, x = torch.meshgrid(torch.arange(1, 65, dtype=torch.float32),
                          torch.arange(1, 65, dtype=torch.float32), indexing="ij")
    index = torch.arange(channels // 2, dtype=torch.float32)
    periods = 10000 ** (2 * (index // 2) / (channels // 2))
    def phase(value):
        t = (value / (64.0 + 1e-6) * (2 * np.pi)).reshape(-1, 1) / periods
        return torch.stack((t[:, ::2].sin(), t[:, 1::2].cos()), dim=-1).flatten(1)
    return torch.cat((phase(y), phase(x)), dim=1)[None]


def constants():
    import torch

    periods = 1 / (10000 ** (torch.arange(0, 256, 4, dtype=torch.float32) / 256))
    sequence = torch.arange(64 * 64)
    phase = torch.cat((torch.outer((sequence % 64).float(), periods),
                       torch.outer((sequence // 64).float(), periods)), dim=1)
    return {"constants.vision_position": spatial_position(256),
            "constants.memory_position": spatial_position(64),
            "constants.rope_cos": phase.cos(), "constants.rope_sin": phase.sin()}


def split(state):
    import torch

    encoder = {k: v for k, v in state.items()
               if k.startswith(("image_encoder.", "sam_mask_decoder.conv_s0.", "sam_mask_decoder.conv_s1."))}
    decoder = {k: v for k, v in state.items()
               if (k.startswith(("sam_prompt_encoder.", "sam_mask_decoder.")) and k not in encoder) or k == "no_mem_embed"}
    # The 14x14 positional background is resized once for the fixed 1024x1024 profile.
    position = "image_encoder.trunk.pos_embed"
    encoder[position] = torch.nn.functional.interpolate(encoder[position], (256, 256), mode="bicubic")
    memory = {k: v for k, v in state.items() if k.startswith(("memory_encoder.", "obj_ptr_proj.")) or k == "no_obj_ptr"}
    attention = {k: v for k, v in state.items() if k.startswith("memory_attention.") or k == "maskmem_tpos_enc"}
    attention.update(constants())
    return encoder, decoder, memory, attention


def state_dict(path):
    from safetensors.torch import load_file

    state = load_file(path, device="cpu")
    if len(state) != 612 or state["image_encoder.trunk.pos_embed"].shape != (1, 112, 14, 14):
        raise ValueError("Expected SAM 2.0 Hiera base+")
    return state


def convert(args):
    from safetensors.torch import save_file

    weights = state_dict(args.checkpoint)
    metadata = {"checkpoint_sha256": sha256(args.checkpoint), "model": "sam2_hiera_base_plus", "precision": "float32"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
        staged = Path(temp) / "weights"
        staged.mkdir()
        for key, tensors in zip(PARTS, split(weights), strict=True):
            save_file({name: tensor.contiguous() for name, tensor in tensors.items()}, staged / f"{key}.safetensors", metadata)
            print(f"{key}: {len(tensors)} tensors")
        staged.rename(args.output)


def reference(node, checkpoint, mode="single_image"):
    import torch

    # Load the custom node's reference modules without importing the ComfyUI server or node registration.
    package = types.ModuleType("zml_sam2_reference")
    package.__path__ = [str(node.resolve())]
    sys.modules[package.__name__] = package
    sys.modules["comfy"] = types.ModuleType("comfy")
    utils = types.ModuleType("comfy.utils")
    utils.load_torch_file = state_dict
    sys.modules["comfy.utils"] = utils
    module = importlib.import_module("zml_sam2_reference.load_model")
    config = node / "sam2_configs" / "sam2_hiera_b+.yaml"
    return module.load_model(checkpoint, config, mode, torch.float32, "cpu")


def validate(args):
    import torch

    torch.set_num_threads(8)
    predictor = reference(args.reference, args.checkpoint)
    model = predictor.model
    args.output.mkdir(parents=True, exist_ok=False)
    source = read_rgb(args.image)
    h, w = source.shape[:2]
    digest = sha256(args.checkpoint)
    weights = state_dict(args.checkpoint)
    encoder, decoder = [args.weights / f"{part}.safetensors" for part in ("encoder", "decoder")]
    for part, tensors in zip(PARTS, split(weights), strict=True):
        if part in ("memory", "attention") and not args.video:
            continue
        part = args.weights / f"{part}.safetensors"
        check_checkpoint(part, tensors, digest)
    del weights, tensors
    with torch.inference_mode():
        data = predictor._transforms(source)[None]
        features = model.forward_image(data)["backbone_fpn"]
    raw = probe(args, "encode", encoder, data.numpy(), (sum(EMBED),), "embedding")
    offset = 0
    metrics = {"features": [], "prompts": []}
    for feature, count in zip((features[2], features[0], features[1]), EMBED, strict=True):
        metrics["features"].append(compare("Hiera feature", raw[offset:offset+count].reshape(feature.shape),
                                           feature.numpy(), 4e-3, 7e-3))
        offset += count
    low = features[2] + model.no_mem_embed.reshape(1, 256, 1, 1)
    previous_mask = None
    for tag in ("points", "box_refine"):
        if tag == "points":
            xy = np.array([[0.5, 0.39], [0.1, 0.1]], dtype=np.float32) * 1024
            labels = [1, 0]
            mask = None
        else:
            xy = np.array([[0.2, 0.2], [0.8, 0.8]], dtype=np.float32) * 1024
            labels = [2, 3]
            mask = previous_mask
        packed = np.column_stack((np.vstack((xy, [0, 0])), labels + [-1]))
        with torch.inference_mode():
            sparse, dense = model.sam_prompt_encoder(
                points=(torch.from_numpy(xy[None]), torch.tensor([labels])), boxes=None, masks=mask)
            expected, scores, tokens, obj = model.sam_mask_decoder.predict_masks(
                image_embeddings=low, image_pe=model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, repeat_image=False,
                high_res_features=features[:2])
            expected_raster = predictor._transforms.postprocess_masks(expected, (h, w))[0].numpy() > 0
        package = np.concatenate((raw, [float(mask is not None)],
                                  mask.numpy().ravel() if mask is not None else np.zeros(256**2, np.float32),
                                  packed.ravel(), [1]))
        pixels = 4 * 256**2
        out = probe(args, "decode", decoder, package, (pixels + 4 + 4 * 256 + 1,), tag, len(packed))
        logits = out[:pixels].reshape(4, 256, 256)
        raster = np.stack([cv2.resize(logit, (w, h)) > 0 for logit in logits])
        result = {"name": tag, "logit_max_abs": compare(tag, logits, expected[0].numpy(), 1e-2, 6e-3),
                  "score_max_abs": compare(tag + " scores", out[pixels:pixels+4], scores[0].numpy(), 3e-3, 4e-3),
                  "token_max_abs": compare(tag + " tokens", out[pixels+4:-1], tokens.numpy().ravel(), 4e-3, 5e-3),
                  "object_max_abs": compare(tag + " object", out[-1:], obj.numpy().ravel(), 5e-3, 5e-3),
                  "min_original_mask_iou": raster_iou(raster, expected_raster)}
        metrics["prompts"].append(result)
        previous_mask = expected[:, 0:1]
    if args.video:
        from video_validation import validate_tracking

        metrics["video"] = validate_tracking(args, source, model, predictor._transforms)
    write_report(args, metrics)


if __name__ == "__main__":
    options = arguments(argparse.ArgumentParser(description=__doc__), video=True)
    (convert if options.action == "convert" else validate)(options)
