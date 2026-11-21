"""Restore faces with the three ZML networks and ComfyUI-compatible OpenCV geometry."""

import argparse
import ctypes as c
import json
from pathlib import Path

import cv2
import numpy as np


BACKENDS = {"cpu": (1, "cpu"), "cuda": (2, "iree_cuda"), "rocm": (3, "iree_rocm"), "vulkan": (4, "iree_vulkan")}
MODELS = {"retinaface": 1, "parsenet": 2, "codeformer": 3}
SIZE = 512
FACE_VALUES = 3 * SIZE * SIZE
P = c.POINTER(c.c_float)
U32 = c.c_uint32
SZ = c.c_size_t
HANDLE = c.c_void_p
TEMPLATE = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                     [201.26117, 371.41043], [313.08905, 371.15118]], dtype=np.float32)
FOREGROUND = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0], dtype=np.float64)


class DeviceInfo(c.Structure):
    _fields_ = [("backend", U32), ("shards", U32), ("pjrt_platform", c.c_char * 32), ("device", c.c_char * 256)]


class ZmlFaces:
    def __init__(self, binary, weights, backend):
        self.backend, self.expected_platform = BACKENDS[backend]
        self.handle = HANDLE()
        self.lib = c.CDLL(str(binary.resolve(strict=True)))
        self._bind("open", U32, U32, c.POINTER(HANDLE))
        self._bind("close", None, HANDLE)
        self._bind("load", U32, HANDLE, U32, c.c_char_p)
        self._bind("infer", U32, HANDLE, U32, U32, U32, c.c_float, P, SZ, P, SZ)
        self._bind("decode", U32, U32, U32, P, SZ, P, SZ, c.POINTER(SZ))
        self._bind("info", U32, HANDLE, c.POINTER(DeviceInfo))
        self._bind("last_error", c.c_char_p)
        self._check(self.open(self.backend, c.byref(self.handle)), "opening session")
        try:
            for model, mid in MODELS.items():
                path = (weights / f"{model}.safetensors").resolve(strict=True)
                self._check(self.load(self.handle, mid, str(path).encode("utf-8")), f"loading {model}")
            self.device_info()
        except BaseException:
            self.close(self.handle)
            raise

    def _bind(self, name, restype, *argtypes):
        function = getattr(self.lib, f"zml_faces_{name}")
        function.argtypes = argtypes
        function.restype = restype
        setattr(self, name, function)

    def _check(self, status, action):
        if status:
            raise RuntimeError(f"ZML {action}: {self.last_error()!r}")

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close(self.handle)
        self.handle = HANDLE()

    def device_info(self, require_placement=False):
        info = DeviceInfo()
        self._check(self.info(self.handle, c.byref(info)), "reading device")
        if info.backend != self.backend or info.pjrt_platform.decode("utf-8") != self.expected_platform:
            raise RuntimeError("ZML selected an unexpected backend")
        if require_placement and not info.shards:
            raise RuntimeError("ZML output device was not verified")
        return {"pjrt": self.expected_platform, "device": info.device.decode("utf-8"), "checked_shards": info.shards}

    def run(self, model, data, shape, fidelity=0):
        data = np.ascontiguousarray(data, dtype=np.float32)
        if data.ndim != 3 or data.shape[0] != 3:
            raise ValueError("Expected a three-channel CHW array")
        _, height, width = data.shape
        output = np.empty(shape, dtype=np.float32)
        code = self.infer(self.handle, MODELS[model], width, height, fidelity, data.ctypes.data_as(P),
                          data.size, output.ctypes.data_as(P), output.size)
        self._check(code, f"running {model}")
        self.device_info(require_placement=True)
        if not np.isfinite(output).all():
            raise ValueError(f"{model} produced nonfinite output")
        return output

    def detect(self, bgr, trace=None):
        height, width = bgr.shape[:2]
        anchors = 2 * sum(((height + d - 1) // d) * ((width + d - 1) // d) for d in (8, 16, 32))
        data = bgr.transpose(2, 0, 1) - np.array([104, 117, 123], dtype=np.float32)[:, None, None]
        raw = self.run("retinaface", data, (anchors, 16))
        out = np.empty((anchors, 15), dtype=np.float32)
        count = SZ()
        status = self.decode(width, height, raw.ctypes.data_as(P), raw.size,
                             out.ctypes.data_as(P), out.size, c.byref(count))
        self._check(status, "decoding RetinaFace")
        if trace is not None:
            trace["retinaface_raw"] = raw
        return out[:count.value]

    def restore(self, crop, fidelity):
        result = self.run("codeformer", normalize(crop), (FACE_VALUES + 256 * 1024,), fidelity)
        return result[:FACE_VALUES].reshape(3, SIZE, SIZE), result[FACE_VALUES:].reshape(256, 1024)

    def parse(self, face):
        logits_rgb = self.run("parsenet", normalize(face), (22, SIZE, SIZE))
        return logits_rgb[:19], logits_rgb[19:]


def normalize(bgr):
    rgb = bgr[:, :, ::-1].astype(np.float32) / np.float32(255)
    return np.ascontiguousarray((rgb.transpose(2, 0, 1) - np.float32(0.5)) / np.float32(0.5))


def restored_bgr(rgb):
    return np.ascontiguousarray(np.rint((np.clip(rgb, -1, 1).transpose(1, 2, 0)[:, :, ::-1] + 1) * 127.5),
                                dtype=np.uint8)


def paste_face(background, face, logits, inverse):
    width, height = background.shape[1], background.shape[0]
    warped = cv2.warpAffine(face, inverse, (width, height))
    square = cv2.warpAffine(np.ones((SIZE, SIZE), dtype=np.float32), inverse, (width, height))
    eroded = cv2.erode(square, np.ones((2, 2), dtype=np.uint8))
    edge = int(np.sum(eroded) ** 0.5) // 20
    center = cv2.erode(eroded, np.ones((edge * 2, edge * 2), dtype=np.uint8))
    soft = cv2.GaussianBlur(center, (edge * 2 + 1, edge * 2 + 1), 0)

    mask = FOREGROUND[np.argmax(logits, axis=0)] * 255
    for _ in range(2):
        mask = cv2.GaussianBlur(mask, (101, 101), 11)
    mask[:10, :] = mask[-10:, :] = mask[:, :10] = mask[:, -10:] = 0
    mask = cv2.warpAffine(mask / 255, inverse, (width, height), flags=cv2.INTER_AREA)
    alpha = np.minimum(soft, mask)[:, :, None]
    return alpha * (eroded[:, :, None] * warped) + (1 - alpha) * background


def restore_image(source, models, fidelity):
    if source.ndim != 3 or source.shape[2] != 3 or source.dtype != np.uint8 or min(source.shape[:2]) == 0:
        raise ValueError("Expected a BGR8 image")
    if not np.isfinite(fidelity) or not 0 <= fidelity <= 1:
        raise ValueError("Fidelity must be between 0 and 1")
    original_h, original_w = source.shape[:2]
    image = source.astype(np.float32)
    if min(original_h, original_w) < SIZE:
        scale = SIZE / min(original_h, original_w)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    height, width = image.shape[:2]
    scale = max(1, 640 / min(height, width))
    detect_input = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_LINEAR)
    detections = models.detect(detect_input)
    background = image.copy()
    count = 0
    for detection in detections:
        points = (detection[5:15] / scale).reshape(5, 2)
        if np.linalg.norm(points[0] - points[1]) < 5:
            continue
        matrix = cv2.estimateAffinePartial2D(points, TEMPLATE, method=cv2.LMEDS)[0]
        if matrix is None:
            continue
        crop = cv2.warpAffine(image, matrix, (SIZE, SIZE), borderValue=(135, 133, 132))
        rgb, _ = models.restore(crop, fidelity)
        face = restored_bgr(rgb)
        logits, _ = models.parse(face)
        background = paste_face(background, face, logits, cv2.invertAffineTransform(matrix))
        count += 1
    output = background.astype(np.uint8)
    if (height, width) != (original_h, original_w):
        output = cv2.resize(output, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
    return output, count


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--dll", type=Path, required=True)
    args.add_argument("--weights", type=Path, required=True)
    args.add_argument("--backend", choices=BACKENDS, required=True)
    args.add_argument("--input", type=Path, required=True)
    args.add_argument("--output", type=Path, required=True)
    args.add_argument("--fidelity", type=float, default=0.5)
    options = args.parse_args()
    if options.output.suffix.lower() != ".png":
        args.error("--output must be a PNG path")
    image = cv2.imdecode(np.fromfile(options.input, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode {options.input}")
    with ZmlFaces(options.dll, options.weights, options.backend) as models:
        output, count = restore_image(image, models, options.fidelity)
        device = models.device_info()
    ok, encoded = cv2.imencode(".png", output)
    if not ok:
        raise ValueError("Could not encode the restored PNG")
    options.output.parent.mkdir(parents=True, exist_ok=True)
    with options.output.open("xb") as stream:
        stream.write(encoded.tobytes())
    print(json.dumps({"faces": count, "output_path": str(options.output.resolve()), **device}))


if __name__ == "__main__":
    main()
