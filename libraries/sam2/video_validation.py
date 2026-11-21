"""Replay deterministic forward tracking with a persistent SAM 2 object memory."""

import cv2
import numpy as np

from segmentation_validation import compare, probe, raster_iou


PRIMARY = 256 * 64**2
EMBEDDINGS = PRIMARY + 32 * 256**2 + 64 * 128**2
MASKS = 4 * 256**2
SPATIAL = 64 * 64**2


def pack_attention(feature, history, frame):
    # One conditioning frame, then the last six propagated frames, oldest first.
    slots = [(6, history[0][1])]
    slots += [(frame - i - 1, memory) for i, memory in history[1:][-6:]]
    past = [history[0][1]] + [memory for _, memory in reversed(history[1:][-15:])]
    return np.concatenate([feature] + [piece for index, memory in slots for piece in ([index], memory[:SPATIAL])]
                          + [entry[SPATIAL:] for entry in past])


def pack_decoder(features, prompts, initial):
    return np.concatenate((features, np.zeros(1 + 256**2, np.float32), prompts.ravel(), [float(initial)]))


def select_mask(result):
    masks = result[:MASKS].reshape(4, 256, 256)
    scores = result[MASKS:MASKS+4]
    tokens = result[MASKS+4:-1].reshape(4, 256)
    chosen = int(np.argmax(scores[1:])) + 1
    obj = result[-1]
    mask = masks[chosen] if obj > 0 else np.full_like(masks[chosen], -1024)
    return mask, tokens[chosen], obj


def validate_tracking(args, source, model, transform):
    import torch

    h, w = source.shape[:2]
    model.binarize_mask_from_pts_for_mem_enc = True
    ref_history = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    history = []
    reports = []
    for frame in range(3):
        initial = frame == 0
        if initial:
            image = source
            coords = np.array([512, 400], dtype=np.float32)
            points = {"point_coords": torch.from_numpy(coords).reshape(1, 1, 2), "point_labels": torch.ones(1, 1, dtype=torch.int32)}
            prompts = np.array([[*coords, 1], [0, 0, -1]], dtype=np.float32)
        else:
            image = cv2.warpAffine(source, np.array([[1, 0, frame * 5], [0, 1, frame * 3]], np.float32),
                                   (w, h), borderMode=cv2.BORDER_REFLECT_101)
            points = None
            prompts = np.array([[0, 0, -1], [0, 0, -1]], dtype=np.float32)
        with torch.inference_mode():
            data = transform(image)[None]
            backbone, flat, pos, sizes = model._prepare_backbone_features(model.forward_image(data))
            reference = model._prepare_memory_conditioned_features(
                frame, initial, flat[-1:], pos[-1:], sizes[-1:], ref_history, num_frames=3)
            _, _, _, ref_mask, ref_high, ref_ptr, obj_ref = model._forward_sam_heads(
                reference, point_inputs=points, high_res_features=backbone["backbone_fpn"][:2], multimask_output=True)
            ref_features, positions = model._encode_new_memory(flat, sizes, ref_high, obj_ref, is_mask_from_pts=initial)
        target = "cond_frame_outputs" if initial else "non_cond_frame_outputs"
        ref_history[target][frame] = {"maskmem_features": ref_features, "maskmem_pos_enc": positions, "obj_ptr": ref_ptr}

        at = f"video-{frame}"
        features = probe(args, "encode", args.weights / "encoder.safetensors", data.numpy(),
                         (EMBEDDINGS,), at + "-embedding")
        if initial:
            decoder_features = features
            memory_error = None
        else:
            package = pack_attention(features[:PRIMARY], history, frame)
            conditioned = probe(args, "memory_attention", args.weights / "attention.safetensors", package,
                                (PRIMARY,), at + "-attention", package.size)
            memory_error = compare(at + " conditioned embedding", conditioned, reference.numpy().ravel(), 1e-2, 1e-2)
            decoder_features = np.concatenate((conditioned, features[PRIMARY:]))
        output = probe(args, "decode", args.weights / "decoder.safetensors",
                       pack_decoder(decoder_features, prompts, initial), (MASKS + 4 + 4 * 256 + 1,),
                       at + "-mask", len(prompts))
        mask, token, obj = select_mask(output)
        logit_error = compare(at + " mask", mask, ref_mask.numpy()[0, 0], 2e-2, 1e-2)
        ref_raster = ref_mask.numpy()[0, 0] > 0
        iou = raster_iou((mask[None] > 0), ref_raster[None])
        if np.count_nonzero(mask > 0) == 0:
            raise AssertionError("Tracking fixture lost the prompted object")
        np.save(args.output / f"{at}-mask.npy", mask)
        high = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        packet = np.concatenate((features[:PRIMARY], high.ravel(), token, [obj, float(initial)]))
        state = probe(args, "memory_encode", args.weights / "memory.safetensors", packet,
                      (SPATIAL + 256,), at + "-memory")
        history.append((frame, state))
        state_error = compare(at + " spatial memory", state[:SPATIAL], ref_features.numpy().ravel(), 4e-2, 1e-2)
        pointer_error = compare(at + " object pointer", state[SPATIAL:], ref_ptr.numpy().ravel(), 1e-2, 1e-2)
        reports.append({"frame": frame, "conditioned_max_abs": memory_error, "mask_max_abs": logit_error,
                        "mask_iou": iou, "memory_max_abs": state_error, "pointer_max_abs": pointer_error})
    return reports
