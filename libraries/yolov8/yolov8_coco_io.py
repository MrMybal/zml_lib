"""Convert the local YOLOv8x COCO detector and stream video frames to ZML."""

from contextlib import contextmanager
import subprocess
import sys

import cv2
import numpy as np

from probe_placement import verify_placement


COCO_FIELDS = 84
NMS_LIMIT = 300
NMS_FIELDS = 6


def convert(probe, checkpoint, weights):
    import torch
    from safetensors.torch import save_file
    from ultralytics import YOLO

    model = YOLO(str(checkpoint)).model.float().eval()
    head = model.model[-1]
    if type(head).__name__ != "Detect" or head.nc != 80 or head.reg_max != 16 or model.names[0] != "person":
        raise ValueError("Expected a YOLOv8x COCO detection checkpoint")
    expected = {}
    for line in subprocess.check_output([probe, "schema", "coco"], text=True).splitlines():
        name, *dims = line.split("\t")
        if name in expected or not dims:
            raise ValueError(f"Invalid detector schema entry {line!r}")
        expected[name] = tuple(map(int, dims))
    state = {name: tensor.detach().contiguous() for name, tensor in model.state_dict().items()
             if not name.endswith(".num_batches_tracked")}
    if state.keys() != expected.keys():
        raise ValueError(f"Mismatched YOLOv8x tensors: {sorted(state.keys() ^ expected.keys())}")
    for name, shape in expected.items():
        if state[name].dtype != torch.float32 or tuple(state[name].shape) != shape:
            raise ValueError(f"Expected FP32 {name} with shape {shape}")
    save_file(state, weights, metadata={"architecture": "yolov8x-coco", "storage": "F32"})
    return model


@contextmanager
def read_frames(video):
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open {video}")

        def frames():
            while True:
                valid, frame = cap.read()
                if not valid:
                    return
                yield frame

        yield frames()
    finally:
        cap.release()


def prepare_video(video, source):
    from ultralytics.data.augment import LetterBox

    letterbox = LetterBox(new_shape=(640, 640), auto=True, stride=32)
    shape = None
    count = 0
    with read_frames(video) as frames, source.open("xb") as stream:
        for frame in frames:
            canvas = letterbox(image=frame)
            if count and canvas.shape[:2] != shape:
                raise ValueError("YOLOv8x video dimensions changed")
            shape = canvas.shape[:2]
            normalize_bgr(canvas).tofile(stream)
            count += 1
    if not count:
        raise ValueError("No video frames for YOLOv8x")
    return count, shape


def normalize_bgr(canvas):
    rgb = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1), dtype="<f4")
    return rgb / np.float32(255)


def write_bgr(frames, source):
    shape = None
    count = 0
    with source.open("xb") as stream:
        for frame in frames:
            if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
                raise ValueError("Expected HWC BGR8 video frames")
            if shape is None:
                shape = frame.shape[:2]
            elif frame.shape[:2] != shape:
                raise ValueError("YOLOv8x video dimensions changed")
            frame.tofile(stream)
            count += 1
    if not 1 <= count <= 4096:
        raise ValueError("Expected 1–4096 video frames for YOLOv8x")
    return count, shape


def letterbox_shape(probe, input_shape):
    h, w = input_shape
    canvas_w, canvas_h = map(int, subprocess.check_output([probe, "letterbox-shape", str(w), str(h)], text=True).split())
    if not all(32 <= side <= 640 and side % 32 == 0 for side in (canvas_h, canvas_w)):
        raise ValueError("Invalid ZML letterbox dimensions")
    return canvas_h, canvas_w


def infer(probe, backend, weights, source, output, count, shape, repeats, expected_device, *, bgr_shape=None):
    input_h, input_w = shape if bgr_shape is None else bgr_shape
    cmd = [probe, *(["bgr"] if bgr_shape is not None else []), backend, "coco", weights,
           str(input_w), str(input_h), source, output, str(count), str(repeats)]
    run_probe(cmd, backend, count * repeats, expected_device)
    cells = (shape[0] // 32) * (shape[1] // 32) * 21
    if output.stat().st_size != count * COCO_FIELDS * cells * 4:
        raise ValueError("Invalid YOLOv8x output file size")
    return (count, COCO_FIELDS, cells)


def run_probe(command, backend, placements, expected_device):
    result = subprocess.run(command, stdout=sys.stderr, stderr=subprocess.PIPE, text=True)
    sys.stderr.write(result.stderr)
    result.check_returncode()
    verify_placement(result.stderr, backend, placements, expected_device)


def infer_nms(probe, backend, weights, source, output, count, canvas, repeats, expected_device, *, bgr_shape,
              raw_output=None, source_coordinates=False):
    h, w = bgr_shape
    mode = "bgr-nms-source" if source_coordinates else "bgr-nms"
    cmd = [probe, mode, backend, "coco", weights, str(w), str(h), source, output, str(count), str(repeats)]
    if raw_output is not None:
        cmd.extend(("--raw-output", raw_output))
    run_probe(cmd, backend, count * repeats * 2, expected_device)
    shape = (count, NMS_LIMIT, NMS_FIELDS)
    if output.stat().st_size != count * NMS_LIMIT * NMS_FIELDS * 4:
        raise ValueError("Invalid YOLOv8x NMS output file size")
    cells = (canvas[0] // 32) * (canvas[1] // 32) * 21
    if raw_output is not None and raw_output.stat().st_size != count * COCO_FIELDS * cells * 4:
        raise ValueError("Invalid YOLOv8x raw validation output file size")
    return shape, cells


def detections(rows, cells):
    if rows.shape != (NMS_LIMIT, NMS_FIELDS) or not np.isfinite(rows).all():
        raise ValueError("Invalid YOLOv8x NMS output")
    anchors = rows[:, 5]
    count = int(np.count_nonzero(anchors >= 0))
    if ((anchors[:count] < 0).any() or (anchors[:count] >= cells).any()
            or (anchors != np.trunc(anchors)).any() or (anchors[count:] != -1).any()
            or np.any(rows[count:, :5]) or (rows[:count, 4] <= 0.5).any()):
        raise ValueError("Invalid YOLOv8x kept anchors or padding")
    indices = anchors[:count].astype(np.int32)
    if np.unique(indices).size != count:
        raise ValueError("YOLOv8x repeated a kept anchor")
    values = rows[:count].copy()
    values[:, 5] = 0  # BoT-SORT expects the person class ID in the last column.
    return values, indices
