"""Run the YOLOv8x graph in ZML and preserve GVHMR's BoT-SORT track selection."""

from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from botsort_io import cost_worker, with_botsort_costs
import yolov8_coco_io as detector


def native_detector(args, work, metrics):
    if not args.yolov8_probe:
        return nullcontext()

    from hmr4d.utils.preproc import tracker

    class NativeTracker(tracker.Tracker):
        def __init__(self):
            pass

        def track(self, video_path):
            from ultralytics.engine.results import Boxes
            from ultralytics.trackers.bot_sort import BOTSORT
            from ultralytics.utils import IterableSimpleNamespace, YAML
            from ultralytics.utils.checks import check_yaml

            source = work / "yolo-input.bgr"
            output = work / "yolo-output.f32"
            audit = work / "yolo-raw.f32" if args.verify else None
            weights = work / "yolov8x.safetensors"
            nc = 80
            checkpoint = Path("inputs/checkpoints/yolo/yolov8x.pt").resolve(strict=True)
            reference = detector.convert(args.yolov8_probe, checkpoint, weights)
            with detector.read_frames(video_path) as frames:
                count, bgr_shape = detector.write_bgr(frames, source)
            shape = detector.letterbox_shape(args.yolov8_probe, bgr_shape)
            repeats = 2 if args.verify else 1
            print(f"YOLOv8x ZML: {count} frames of {shape[1]}×{shape[0]}", flush=True)
            _, cells = detector.infer_nms(args.yolov8_probe, args.backend, weights, source, output, count, shape,
                                          repeats, args.expected_device, bgr_shape=bgr_shape, raw_output=audit,
                                          source_coordinates=True)
            cfg = IterableSimpleNamespace(**YAML.load(check_yaml("botsort.yaml")))
            worker = (cost_worker(args.botsort_cost_probe, args.backend, work, expected_device=args.expected_device)
                      if args.botsort_cost_probe else nullcontext())
            box_error = score_error = 0.0
            kept_count = 0
            letterbox = None
            if args.verify:
                import torchvision  # Select Ultralytics' stable CPU NMS reference.
                from ultralytics.data.augment import LetterBox
                from ultralytics.utils.nms import non_max_suppression
                from ultralytics.utils.ops import scale_boxes

                letterbox = LetterBox(new_shape=(640, 640), auto=True, stride=32)
                reference.cuda()
            history = []
            with (worker as costs, detector.read_frames(video_path) as frames, output.open("rb") as predictions,
                  (audit.open("rb") if audit is not None else nullcontext()) as raw,
                  torch.backends.cudnn.flags(allow_tf32=False)):
                tracker_type = BOTSORT if costs is None else with_botsort_costs(costs, verify=args.verify)
                tracker_instance = tracker_type(args=cfg, frame_rate=30)
                for frame in frames:
                    if frame.shape[:2] != bgr_shape:
                        raise ValueError("Decoded frame size changed during YOLOv8x tracking")
                    rows = np.fromfile(predictions, dtype="<f4", count=detector.NMS_LIMIT * detector.NMS_FIELDS)
                    kept, indices = detector.detections(rows.reshape(detector.NMS_LIMIT, detector.NMS_FIELDS), cells)
                    kept_count += len(kept)
                    if letterbox is not None:
                        values = np.fromfile(raw, dtype="<f4", count=detector.COCO_FIELDS * cells).reshape(detector.COCO_FIELDS, cells)
                        if not np.isfinite(values).all():
                            raise ValueError("YOLOv8x returned nonfinite predictions")
                        image = detector.normalize_bgr(letterbox(image=frame))
                        if image.shape != (3, *shape):
                            raise ValueError("The Ultralytics reference returned a different letterbox shape")
                        expected, _ = reference(torch.from_numpy(image)[None].cuda())
                        expected = expected[0].cpu().numpy()
                        np.testing.assert_allclose(values[:4], expected[:4], atol=0.03, rtol=2e-4)
                        np.testing.assert_allclose(values[4:], expected[4:], atol=2e-3, rtol=2e-3)
                        box_error = max(box_error, float(np.abs(values[:4] - expected[:4]).max()))
                        score_error = max(score_error, float(np.abs(values[4:] - expected[4:]).max()))
                        reference_kept, reference_indices = non_max_suppression(
                            torch.from_numpy(values)[None], 0.5, 0.7, classes=[0], nc=nc, return_idxs=True)
                        reference_kept[0][:, :4] = scale_boxes(shape, reference_kept[0][:, :4], frame.shape[:2])
                        np.testing.assert_array_equal(indices, reference_indices[0].numpy().reshape(-1))
                        np.testing.assert_array_equal(kept, reference_kept[0].numpy())
                    detections = torch.from_numpy(kept)
                    tracks = tracker_instance.update(Boxes(detections, frame.shape[:2]).numpy(), frame)
                    history.append([{"id": int(track[4]), "bbx_xyxy": track[:4]} for track in tracks])
            if len(history) != count:
                raise ValueError("Decoded frame count changed during YOLOv8x inference")
            metrics.update(yolo_engine="zml", yolo_preprocess_engine="zml", yolo_dispatches=count * repeats,
                           yolo_nms_engine="zml", yolo_nms_dispatches=count * repeats, yolo_kept_person_boxes=kept_count,
                           yolo_projection_engine="zml",
                           yolo_tracked_frames=sum(bool(tracks) for tracks in history))
            if args.verify:
                metrics.update(yolo_max_abs_xywh=box_error, yolo_max_abs_score=score_error, yolo_nms_identical=True)
            if costs is not None and costs.dispatches:
                metrics.update(botsort_cost_engine="zml", botsort_cost_dispatches=costs.dispatches,
                               botsort_cost_stage_dispatches=dict(costs.stage_dispatches),
                               botsort_cost_capacities=costs.capacities)
            return history

    return patch.object(tracker, "Tracker", NativeTracker)
