"""Convert SAM ViT-B and compare two prompt sets with the same ZML image embedding."""

import argparse
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from segmentation_validation import arguments, check_checkpoint, compare, probe, raster_iou, read_rgb, sha256, write_report


def split(state):
    return ({k: v for k, v in state.items() if k.startswith("image_encoder.")},
            {k: v for k, v in state.items() if not k.startswith("image_encoder.")})


def convert(args):
    import torch
    from safetensors.torch import save_file

    weights = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if len(weights) != 314 or weights["image_encoder.pos_embed"].shape != (1, 64, 64, 768):
        raise ValueError("Expected SAM ViT-B at image size 1024")
    metadata = {"checkpoint_sha256": sha256(args.checkpoint), "model": "sam_vit_b", "precision": "float32"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
        staged = Path(temp) / "weights"
        staged.mkdir()
        for key, state in zip(("encoder", "decoder"), split(weights), strict=True):
            save_file({name: tensor.contiguous() for name, tensor in state.items()}, staged / f"{key}.safetensors", metadata)
            print(f"{key}: {len(state)} tensors")
        staged.rename(args.output)


def validate(args):
    import torch

    torch.set_num_threads(8)
    # The path contains the installed segment_anything package from the ComfyUI node.
    sys.path.insert(0, str(args.reference.resolve()))
    from segment_anything import sam_model_registry
    from segment_anything.utils.transforms import ResizeLongestSide

    args.output.mkdir(parents=True, exist_ok=False)
    source = read_rgb(args.image)
    h, w = source.shape[:2]
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    ref = sam_model_registry["vit_b"](checkpoint=None).eval()
    ref.load_state_dict(state, strict=True)
    digest = sha256(args.checkpoint)
    encoder, decoder = [args.weights / f"{part}.safetensors" for part in ("encoder", "decoder")]
    for part, weights in zip((encoder, decoder), split(state), strict=True):
        check_checkpoint(part, weights, digest)
    transform = ResizeLongestSide(1024)
    pixels = transform.apply_image(source)
    ih, iw = pixels.shape[:2]
    with torch.inference_mode():
        data = ref.preprocess(torch.from_numpy(pixels).permute(2, 0, 1).contiguous()[None])
        embedded = ref.image_encoder(data)
    actual = probe(args, "encode", encoder, data.numpy(), (1, 256, 64, 64), "embedding")
    metrics = {"embedding_max_abs": compare("ViT-B embedding", actual, embedded.numpy(), 3e-3, 5e-3), "prompts": []}

    previous_mask = None
    for tag in ("points", "box_refine"):
        points, box, mask = None, None, None
        if tag == "points":
            xy = transform.apply_coords(np.array([[w * 0.5, h * 0.39], [w * 0.1, h * 0.1]]), (h, w)).astype(np.float32)
            labels = np.array([1, 0])
            points = (torch.from_numpy(xy[None]), torch.from_numpy(labels[None]))
            packed = np.column_stack((np.vstack((xy, [0, 0])), [1, 0, -1]))
        else:
            xy = transform.apply_boxes(np.array([[w * 0.2, h * 0.2, w * 0.8, h * 0.8]]), (h, w)).astype(np.float32)
            box = torch.from_numpy(xy)
            packed = np.column_stack((xy.reshape(2, 2), [2, 3]))
            mask = previous_mask
        with torch.inference_mode():
            sparse, dense = ref.prompt_encoder(points=points, boxes=box, masks=mask)
            expected_mask, expected_iou = ref.mask_decoder.predict_masks(
                image_embeddings=embedded, image_pe=ref.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense)
            expected_raster = ref.postprocess_masks(expected_mask, (ih, iw), (h, w))[0].numpy() > 0
        package = np.concatenate((actual.ravel(), [float(mask is not None)],
                                  mask.numpy().ravel() if mask is not None else np.zeros(256**2, np.float32),
                                  packed.ravel()))
        out = probe(args, "decode", decoder, package, (4 * 256**2 + 4,), tag, len(packed))
        logits = out[:-4].reshape(4, 256, 256)
        raster = np.stack([cv2.resize(cv2.resize(logit, (1024, 1024))[:ih, :iw], (w, h)) > 0 for logit in logits])
        result = {"name": tag, "logit_max_abs": compare(tag, logits, expected_mask[0].numpy(), 1e-2, 4e-3),
                  "score_max_abs": compare(tag + " scores", out[-4:], expected_iou[0].numpy(), 2e-3, 3e-3),
                  "min_original_mask_iou": raster_iou(raster, expected_raster)}
        metrics["prompts"].append(result)
        previous_mask = expected_mask[:, 0:1]
    write_report(args, metrics)


if __name__ == "__main__":
    options = arguments(argparse.ArgumentParser(description=__doc__))
    (convert if options.action == "convert" else validate)(options)
