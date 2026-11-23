"""Connect ZML pose and appearance probes to the pinned GVHMR crop pipeline."""

from contextlib import ExitStack, contextmanager
from functools import lru_cache, partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import crop_io
import pose_checkpoint


ASSETS = {
    "vitpose": Path("inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth"),
    "hmr2": Path("inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt"),
}


def heatmaps_to_joints(heatmaps, boxes):
    from hmr4d.utils.kpts.kp2d_utils import keypoints_from_heatmaps

    center = boxes[:, :2].numpy()
    scale = torch.cat((boxes[:, [2]] * 0.75, boxes[:, [2]]), dim=1).numpy() / 200
    joints, confidence = keypoints_from_heatmaps(heatmaps=heatmaps, center=center, scale=scale, use_udp=True)
    return np.concatenate((joints, confidence), axis=-1)


def joint_error(actual, expected, tolerance):
    # Below-zero scores mark absent joints. The reference's refinement can read
    # a neighboring heatmap for these joints because it starts at (-1, -1).
    visible = (actual[..., 2] > 0) & (expected[..., 2] > 0)
    values, target = actual[visible, :2], expected[visible, :2]
    np.testing.assert_allclose(values, target, atol=tolerance, rtol=0, equal_nan=False)
    return float(np.max(np.abs(values - target), initial=0))


def reference(kind, checkpoint, crops):
    if kind == "vitpose":
        from hmr4d.utils.geo.flip_utils import flip_heatmap_coco17
        from hmr4d.utils.preproc.vitpose_pytorch import build_model

        model = build_model("ViTPose_huge_coco_256x192", str(checkpoint))
    else:
        from hmr4d.network.hmr2 import load_hmr2

        model = load_hmr2(checkpoint)

    model = model.eval().cuda()
    output = []
    with torch.inference_mode(), torch.backends.cudnn.flags(allow_tf32=False):
        for batch in crops.split(4):
            batch = batch.cuda()
            if kind == "vitpose":
                direct, flipped = model(torch.cat((batch, batch.flip(3)), dim=0)).chunk(2)
                value = (direct + flip_heatmap_coco17(flipped)) * 0.5
            else:
                value = model({"img": torch.nn.functional.pad(batch, (32, 32))})
            output.append(value.cpu())
    return torch.cat(output, dim=0).numpy()


def host_crops(video_path, boxes):
    from hmr4d.utils.preproc.vitfeat_extractor import get_batch

    images, cropped_boxes = get_batch(video_path, boxes)
    crops = images[:, :, :, 32:224].contiguous()
    return crops, cropped_boxes, crops


def shared_native_crops(args, work, metrics, frames):
    from hmr4d.utils.preproc.vitfeat_extractor import get_batch
    from hmr4d.utils.video_io_utils import read_video_np

    @lru_cache(maxsize=1)
    def prepare(video_path, boxes):
        height, width = frames[0].shape[:2]
        print(f"Pose resize and crop ZML: {len(frames)} RGB frames of {width}x{height}", flush=True)
        values, half, dispatches, antialiased = crop_io.run_half(
            args.crop_probe, args.backend, frames, boxes.numpy(), work, capture_half=args.verify,
            repeats=2 if args.verify else 1, expected_device=args.expected_device,
        )
        crops = torch.from_numpy(values)
        # GVHMR uses an unchanged full-image crop box to decode the keypoints.
        cropped_boxes = (boxes * 0.5) / 0.5
        metrics.update(pose_resize_engine="zml", pose_resize_input_hw=[height, width],
                       pose_crop_engine="zml", pose_crop_dispatches=dispatches, pose_crop_frames=len(frames),
                       pose_crop_antialiased_frames=antialiased)
        expected = None
        if args.verify:
            reference_half = read_video_np(video_path, scale=0.5)
            np.testing.assert_array_equal(half, reference_half)
            metrics.update(pose_resize_rgb_values=half.size, pose_resize_output_hw=list(half.shape[1:3]),
                           pose_resize_max_abs_error=0, pose_resize_outputs=dispatches)
            reference_images, reference_boxes = get_batch(reference_half, boxes * 0.5, img_ds=1.0, path_type="np")
            expected = reference_images[:, :, :, 32:224].contiguous()
            np.testing.assert_allclose(values, expected.numpy(), atol=1e-6, rtol=0, equal_nan=False)
            np.testing.assert_array_equal(cropped_boxes.numpy(), (reference_boxes / 0.5).numpy())
            metrics["pose_crop_max_abs_error"] = float(np.max(np.abs(values - expected.numpy())))
        return crops, cropped_boxes, expected

    return prepare


def preprocessor(kind, probe, backend, work, expected_device, verify, metrics, crops_for):
    def extract(video_path, boxes):
        crops, cropped_boxes, reference_crops = crops_for(video_path, boxes)
        checkpoint = ASSETS[kind].resolve(strict=True)
        weights = work / f"{kind}.safetensors"
        print(f"{kind} ZML: converting {checkpoint.name} for {len(crops)} crops", flush=True)
        pose_checkpoint.convert(probe, kind, checkpoint, weights)
        print(f"{kind} ZML: running inference", flush=True)
        use_udp = kind == "vitpose"
        values, dispatches = pose_checkpoint.run(
            probe, kind, backend, weights, crops.numpy(), work, repeats=2 if verify else 1,
            expected_device=expected_device, boxes=cropped_boxes.numpy() if use_udp else None,
            keep_heatmaps=use_udp and verify,
        )
        metrics[f"{kind}_engine"] = "zml"
        metrics[f"{kind}_dispatches"] = dispatches
        if use_udp:
            metrics["vitpose_udp_engine"] = "zml"
        result = values
        if verify:
            if use_udp:
                fields = int(np.prod(pose_checkpoint.JOINTS))
                result = values[:, :fields].reshape(len(crops), *pose_checkpoint.JOINTS).copy()
                values = values[:, fields:].reshape(len(crops), *pose_checkpoint.OUTPUTS[kind])
                decoded = heatmaps_to_joints(values, cropped_boxes)
                np.testing.assert_array_equal(result[..., 2], decoded[..., 2])
                metrics["vitpose_udp_max_abs_error_px_same_heatmaps"] = joint_error(result, decoded, 0.02)
                metrics["vitpose_udp_visible_joints"] = int(np.count_nonzero(result[..., 2] > 0))
            print(f"{kind}: comparing with PyTorch FP32", flush=True)
            expected = reference(kind, checkpoint, reference_crops)
            np.testing.assert_allclose(values, expected, atol=8e-4, rtol=2e-3, equal_nan=False, err_msg=kind)
            metrics[f"{kind}_max_abs_error"] = float(np.max(np.abs(values - expected)))
            if use_udp:
                expected_joints = heatmaps_to_joints(expected, cropped_boxes)
                metrics["vitpose_joints_max_abs_error_px"] = joint_error(result, expected_joints, 0.12)
        return torch.from_numpy(result)

    return extract


@contextmanager
def native_pose(args, work, metrics, frames):
    with ExitStack() as stack:
        crops_for = shared_native_crops(args, work, metrics, frames) if args.crop_probe else host_crops
        if args.vitpose_probe:
            from hmr4d.utils.preproc import vitpose

            run = preprocessor("vitpose", args.vitpose_probe, args.backend, work, args.expected_device, args.verify, metrics, crops_for)
            stack.enter_context(patch.object(vitpose, "VitPoseExtractor", partial(SimpleNamespace, extract=run)))
        if args.hmr2_probe:
            from hmr4d.utils.preproc import vitfeat_extractor

            run = preprocessor("hmr2", args.hmr2_probe, args.backend, work, args.expected_device, args.verify, metrics, crops_for)
            stack.enter_context(patch.object(vitfeat_extractor, "Extractor", partial(SimpleNamespace, extract_video_features=run)))
        yield
