"""Convert trusted local YOLOv8 weights and compare Windows ZML inference to Ultralytics."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np


FILES = {"face": "bbox/face_yolov8m.pt", "hand": "bbox/hand_yolov8s.pt", "person": "segm/person_yolov8m-seg.pt"}
PLATFORMS = {"cpu": "cpu", "cuda": "iree_cuda", "vulkan": "iree_vulkan", "rocm": "iree_rocm"}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_model(path, profile):
    import torch
    import ultralytics  # Registers classes used by the explicitly supplied Ultralytics checkpoint.

    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = (payload.get("ema") or payload["model"]).float().eval()
    head = model.model[-1]
    expected = "Segment" if profile == "person" else "Detect"
    if type(head).__name__ != expected or head.nc != 1 or head.reg_max != 16 or model.names[0] != profile:
        raise ValueError(f"Unexpected YOLOv8 architecture for {profile}")
    return model, {"model": profile, "checkpoint_sha256": sha256(path), "ultralytics": ultralytics.__version__, "precision": "float32"}


def convert(models, output):
    from safetensors.torch import save_file

    output.mkdir(parents=True, exist_ok=False)
    for profile, relative in FILES.items():
        net, metadata = load_model(models / relative, profile)
        state = {name: tensor.detach().contiguous()
                 for name, tensor in net.state_dict().items() if not name.endswith(".num_batches_tracked")}
        save_file(state, output / f"{profile}.safetensors", metadata)
        print(f"{profile}: {len(state)} FP32 tensors")


def prepare(image, size):
    from ultralytics.data.augment import LetterBox

    canvas = LetterBox(new_shape=(size, size), auto=False)(image=image)
    return np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / np.float32(255)


def postprocess(prediction, prototype, source_size, network_size, confidence):
    # Single class for the three local profiles. NMS is implemented here to keep the actual path independent.
    valid = prediction[4] > confidence
    rows = prediction[:, valid].T
    rows = rows[np.argsort(-rows[:, 4], kind="stable")[:30_000]]
    xy = rows[:, :2]
    wh = rows[:, 2:4]
    boxes = np.column_stack((xy - wh / 2, xy + wh / 2))
    kept = []
    remaining = np.arange(len(rows))
    area = np.prod(np.maximum(0, boxes[:, 2:] - boxes[:, :2]), axis=1)
    while remaining.size and len(kept) < 300:
        head, remaining = remaining[0], remaining[1:]
        kept.append(head)
        lo = np.maximum(boxes[head, :2], boxes[remaining, :2])
        hi = np.minimum(boxes[head, 2:], boxes[remaining, 2:])
        intersection = np.prod(np.maximum(0, hi - lo), axis=1)
        remaining = remaining[intersection / (area[head] + area[remaining] - intersection + 1e-7) <= 0.45]
    boxes = boxes[kept]
    height, width = source_size
    gain = min(network_size / height, network_size / width)
    pad_x = round((network_size - round(width * gain)) / 2 - 0.1)
    pad_y = round((network_size - round(height * gain)) / 2 - 0.1)
    boxes = (boxes - np.array([pad_x, pad_y, pad_x, pad_y], dtype=np.float32)) / gain
    boxes = np.clip(boxes, 0, np.array([width, height, width, height], dtype=np.float32))
    if prototype is None:
        return boxes, rows[kept, 4], None
    ph, pw = prototype.shape[-2:]
    logits = (rows[kept, 5:] @ prototype.reshape(32, -1)).reshape(-1, ph, pw)
    proto_gain = min(ph / height, pw / width)
    pad = (pw - width * proto_gain, ph - height * proto_gain)
    left, top = (round(p / 2 - 0.1) for p in pad)
    right, bottom = (pw - round(pad[0] / 2 + 0.1), ph - round(pad[1] / 2 + 0.1))
    masks = []
    corners = (np.rint(boxes) if len(boxes) < 50 else np.ceil(boxes)).astype(np.int32)
    for logit, (x0, y0, x1, y1) in zip(logits, corners, strict=True):
        full = cv2.resize(logit[top:bottom, left:right], (width, height), interpolation=cv2.INTER_LINEAR)
        mask = np.zeros((height, width), dtype=bool)
        mask[y0:y1, x0:x1] = full[y0:y1, x0:x1] > 0
        masks.append(mask)
    return boxes, rows[kept, 4], np.asarray(masks, dtype=bool).reshape(-1, height, width)


def compare(args):
    import torch
    from safetensors import safe_open
    from ultralytics.utils import nms, ops

    torch.set_num_threads(6)
    args.output.mkdir(parents=True, exist_ok=False)
    image = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid input image")
    net, metadata = load_model(args.models / FILES[args.profile], args.profile)
    weights = args.weights / f"{args.profile}.safetensors"
    with safe_open(weights, framework="pt", device="cpu") as converted:
        if converted.metadata() != metadata:
            raise ValueError("Converted checkpoint does not match the original")
        state = {name: t for name, t in net.state_dict().items() if not name.endswith(".num_batches_tracked")}
        if set(converted.keys()) != set(state) or any(not torch.equal(converted.get_tensor(name), t) for name, t in state.items()):
            raise ValueError("Converted checkpoint tensor content differs")
    data = prepare(image, args.size)
    src, dst = (args.output / name for name in ("input.f32", "output.f32"))
    data.tofile(src)
    with torch.inference_mode():
        prediction, aux = net(torch.from_numpy(data[None]))
    expected = prediction[0].numpy()
    proto = aux[-1][0].numpy() if args.profile == "person" else None
    run = subprocess.run([args.probe.resolve(), args.backend, args.profile, weights.resolve(),
                          str(args.size), str(args.size), src.resolve(), dst.resolve()], capture_output=True)
    sys.stdout.buffer.write(run.stdout)
    sys.stderr.buffer.write(run.stderr)
    run.check_returncode()
    if f"backend={args.backend} pjrt={PLATFORMS[args.backend]}".encode() not in run.stderr:
        raise AssertionError("Unexpected PJRT backend")
    raw = np.fromfile(dst, dtype="<f4")
    actual = raw[:expected.size].reshape(expected.shape)
    np.testing.assert_allclose(actual[:4], expected[:4], atol=0.03, rtol=2e-4)
    np.testing.assert_allclose(actual[4:], expected[4:], atol=2e-3, rtol=2e-3)
    zml_proto = raw[expected.size:].reshape(proto.shape) if proto is not None else None
    if proto is not None:
        np.testing.assert_allclose(zml_proto, proto, atol=2e-3, rtol=2e-3)
    boxes, scores, masks = postprocess(actual, zml_proto, image.shape[:2], args.size, args.confidence)
    with torch.inference_mode():
        reference = nms.non_max_suppression(prediction.clone(), args.confidence, 0.45, nc=1)[0]
        reference[:, :4] = ops.scale_boxes(data.shape[-2:], reference[:, :4], image.shape[:2])
        if proto is not None and len(reference):
            actual_masks = ops.process_mask_native(torch.from_numpy(proto), reference[:, 6:], reference[:, :4], image.shape[:2])
            expected_masks = actual_masks.numpy().astype(bool)
    if len(reference) == 0:
        raise AssertionError("Fixture did not produce a detection")
    np.testing.assert_allclose(boxes, reference[:, :4].numpy(), atol=0.08, rtol=0)
    np.testing.assert_allclose(scores, reference[:, 4].numpy(), atol=2e-3, rtol=0)
    result = {"profile": args.profile, "backend": args.backend, "pjrt": PLATFORMS[args.backend], "count": len(boxes),
              "checkpoint_sha256": metadata["checkpoint_sha256"], "image_sha256": sha256(args.image),
              "ultralytics": metadata["ultralytics"], "torch_version": str(torch.__version__),
              "max_abs_xywh": float(np.max(np.abs(actual[:4] - expected[:4]))),
              "max_abs_score": float(np.max(np.abs(actual[4] - expected[4])))}
    if proto is not None:
        intersection = (masks & expected_masks).sum(axis=(1, 2))
        union = (masks | expected_masks).sum(axis=(1, 2))
        iou = np.divide(intersection, union, out=np.ones_like(union, dtype=float), where=union > 0)
        if iou.min() < 0.99:
            raise AssertionError(f"Mask IoU too low: {iou}")
        result["min_mask_iou"] = float(iou.min())
        result["max_abs_prototype"] = float(np.max(np.abs(zml_proto - proto)))
        np.save(args.output / "masks.npy", masks)
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="action", required=True)
    c = commands.add_parser("convert")
    c.add_argument("--models", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    v = commands.add_parser("compare")
    for key in ("models", "weights", "image", "probe", "output"):
        v.add_argument(f"--{key}", type=Path, required=True)
    v.add_argument("--profile", choices=FILES, required=True)
    v.add_argument("--backend", choices=PLATFORMS, default="cpu")
    v.add_argument("--size", type=int, default=640)
    v.add_argument("--confidence", type=float, default=0.25)
    args = p.parse_args()
    if args.action == "convert":
        convert(args.models, args.output)
    else:
        compare(args)


if __name__ == "__main__":
    main()
