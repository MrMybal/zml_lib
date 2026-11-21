"""Compare the Windows ZML face DLL and composite with the installed ComfyUI reference."""

import argparse
from contextlib import ExitStack
import json
import logging
from pathlib import Path
import sys
import types
from unittest import mock

import cv2
import numpy as np
from safetensors import safe_open

from checkpoints import inference_state, read_original, sha256
from restore import BACKENDS, ZmlFaces, restore_image


def reference_modules(node):
    import torch

    # Import the local network sources without starting ComfyUI or invoking a downloader.
    sys.path.insert(0, str(node))
    for package in ("basicsr", "basicsr.archs", "basicsr.utils"):
        module = types.ModuleType(package)
        module.__path__ = [str(node.joinpath(*package.split(".")))]
        sys.modules[package] = module
    sys.modules["basicsr.utils"].get_root_logger = logging.getLogger
    comfy = types.ModuleType("comfy")
    comfy.model_management = types.SimpleNamespace(get_torch_device=lambda: torch.device("cpu"))
    sys.modules["comfy"] = comfy
    sys.modules["folder_paths"] = types.ModuleType("folder_paths")

    from basicsr.archs.codeformer_arch import CodeFormer
    from facelib.detection.retinaface.retinaface import RetinaFace
    from facelib.parsing.parsenet import ParseNet
    from facelib.utils import face_restoration_helper

    return face_restoration_helper, {
        "retinaface": RetinaFace(),
        "parsenet": ParseNet(in_size=512, out_size=512, parsing_ch=19),
        "codeformer": CodeFormer(),
    }


def load_reference(node, originals, converted):
    helper, nets = reference_modules(node)
    exact_weights = {}
    for name, model in nets.items():
        raw, path, group = read_original(name, originals)
        state = inference_state(raw, name)
        metadata = {"model": name, "checkpoint_key": group, "checkpoint_sha256": sha256(path), "precision": "float32"}
        with safe_open(converted / f"{name}.safetensors", framework="pt", device="cpu") as saved:
            if saved.metadata() != metadata or set(saved.keys()) != set(state):
                raise AssertionError(f"{name} provenance or tensor names differ")
            for key, tensor in state.items():
                if not np.array_equal(saved.get_tensor(key).numpy(), tensor.numpy()):
                    raise AssertionError(f"{name} converted tensor differs for {key}")
        model.load_state_dict(raw, strict=True)
        model.eval().float()
        exact_weights[name] = metadata["checkpoint_sha256"]
    with ExitStack() as patches:
        patches.enter_context(mock.patch.object(helper, "init_detection_model", return_value=nets["retinaface"]))
        patches.enter_context(mock.patch.object(helper, "init_parsing_model", return_value=nets["parsenet"]))
        instance = helper.FaceRestoreHelper(1, face_size=512, det_model="retinaface_resnet50", use_parse=True, device="cpu")
    return helper, instance, nets, exact_weights


def reference_chain(helper_module, helper, nets, source, fidelity):
    import torch

    trace = {"faces": []}
    reference_detect = nets["retinaface"].detect_faces

    def detect_hook(_module, _inputs, result):
        trace["retinaface_raw"] = torch.cat(result, dim=-1)[0].cpu().numpy()

    def detect(bgr):
        result = reference_detect(bgr)
        trace["det_input"] = bgr
        trace["detections"] = result
        return result

    def parse_hook(_module, _inputs, result):
        for face in trace["faces"]:
            if "parse_logits" not in face:
                face["parse_logits"] = result[0][0].cpu().numpy()
                face["parse_rgb"] = result[1][0].cpu().numpy()
                break

    d = nets["retinaface"].register_forward_hook(detect_hook)
    p = nets["parsenet"].register_forward_hook(parse_hook)
    helper.clean_all()
    try:
        with torch.inference_mode(), mock.patch.object(nets["retinaface"], "detect_faces", side_effect=detect):
            helper.read_image(source.astype(np.float32))
            helper.get_face_landmarks_5(resize=640, eye_dist_threshold=5)
            helper.align_warp_face()
            for crop in helper.cropped_faces:
                x = helper_module.img2tensor(crop / 255, bgr2rgb=True, float32=True)
                x = helper_module.normalize(x, (0.5,) * 3, (0.5,) * 3, inplace=True).unsqueeze(0)
                rgb, code_logits, _ = nets["codeformer"](x, w=fidelity)
                face = np.rint((rgb[0].clamp(-1, 1).cpu().numpy().transpose(1, 2, 0)[:, :, ::-1] + 1) / 2 * 255)
                restored = np.ascontiguousarray(face, dtype=np.uint8)
                trace["faces"].append({"crop": crop, "input": x, "restored": restored,
                                       "restore_rgb": rgb[0].numpy(), "codes": code_logits[0].numpy()})
                helper.add_restored_face(restored)
            helper.get_inverse_affine()
            output = helper.paste_faces_to_input_image()
            if source.shape != output.shape:
                output = cv2.resize(output, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_LINEAR)
            return output, trace
    finally:
        d.remove()
        p.remove()
        helper.clean_all()


def compare(name, actual, expected, atol, rtol=0):
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError(f"{name}: incompatible or nonfinite values")
    delta = actual.astype(np.float64) - expected
    metrics = {"max_abs": float(np.max(np.abs(delta), initial=0)), "rmse": float(np.sqrt(np.mean(delta ** 2))) if delta.size else 0}
    print(f"{name} {metrics}", flush=True)
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=name)
    return metrics


def pixel_metrics(actual, expected):
    if actual.shape != expected.shape:
        raise AssertionError("Composite dimensions differ")
    delta = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    return {"max_abs": int(delta.max()), "mae": float(delta.mean()), "over_2": float(np.mean(delta > 2))}


def save_png(path, bgr):
    ok, encoded = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Could not encode validation PNG")
    path.write_bytes(encoded.tobytes())


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--node", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(6)
    cv2.setNumThreads(2)
    source = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if source is None:
        raise ValueError(f"Could not read {args.image}")
    helper_module, helper, nets, exact_weights = load_reference(args.node.resolve(), args.models, args.weights)
    print("Reference models loaded from local checkpoints", flush=True)
    expected, reference = reference_chain(helper_module, helper, nets, source, 0.5)
    if not reference["faces"]:
        raise AssertionError("Expected a portrait with at least one face")
    save_png(args.output / "reference.png", expected)
    metrics = {}
    with ZmlFaces(args.dll, args.weights, args.backend) as models:
        detect_trace = {}
        detections = models.detect(reference["det_input"], detect_trace)
        metrics["retinaface_raw"] = compare("RetinaFace raw", detect_trace["retinaface_raw"], reference["retinaface_raw"], 2e-3, 2e-3)
        metrics["detections"] = compare("RetinaFace decoded", detections, reference["detections"], 3e-3)
        metrics["faces"] = []
        for face in reference["faces"]:
            rgb, logits = models.restore(face["crop"], 0.5)
            stage = {
                "codeformer_rgb": compare("CodeFormer RGB", rgb, face["restore_rgb"], 2e-3, 2e-3),
                "codeformer_logits": compare("CodeFormer logits", logits, face["codes"], 2e-3, 2e-3),
            }
            np.testing.assert_array_equal(logits.argmax(axis=1), face["codes"].argmax(axis=1))
            parsing, parse_rgb = models.parse(face["restored"])
            stage["parsenet_logits"] = compare("ParseNet logits", parsing, face["parse_logits"], 2e-3, 2e-3)
            stage["parsenet_rgb"] = compare("ParseNet RGB", parse_rgb, face["parse_rgb"], 2e-3, 2e-3)
            with torch.inference_mode():
                zero_reference = nets["codeformer"](face["input"], w=0)[0][0].numpy()
            zero, _ = models.restore(face["crop"], 0)
            stage["codeformer_w0"] = compare("CodeFormer w=0", zero, zero_reference, 2e-3, 2e-3)
            if np.allclose(zero, rgb):
                raise AssertionError("The fidelity parameter had no effect")
            metrics["faces"].append(stage)
        output, count = restore_image(source, models, 0.5)
        if count != len(reference["faces"]):
            raise AssertionError(f"Chain detected {count} faces, expected {len(reference['faces'])}")
        save_png(args.output / "zml.png", output)
        metrics["composite"] = pixel_metrics(output, expected)
        print(f"Full composite {metrics['composite']}", flush=True)
        if metrics["composite"]["mae"] > 0.15 or metrics["composite"]["over_2"] > 0.002:
            raise AssertionError("Composite differs from the ComfyUI output")
        negative = np.full((233, 235, 3), 90, dtype=np.uint8)
        unchanged, n = restore_image(negative, models, 0.5)
        if n:
            raise AssertionError("Unexpected face in the constant image")
        np.testing.assert_array_equal(negative, unchanged)
        metrics["no_face"] = True
        device = models.device_info(require_placement=True)
    report = {"backend": args.backend, **device, "face_count": count, "image_sha256": sha256(args.image),
              "checkpoint_sha256": exact_weights, "torch_version": str(torch.__version__),
              "opencv_version": cv2.__version__, "metrics": metrics}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
