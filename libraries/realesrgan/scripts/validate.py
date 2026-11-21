"""Generate PyTorch reference tensors and verify the ZML CLI and C ABI."""

import argparse
import ctypes as c
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
from safetensors import safe_open

from checkpoints import MODELS, PROVENANCE, RUNTIME_VARIANTS, checkpoint_state, expected_shapes, read_checkpoint, sha256


TAPS = ("conv_first", "body.0.rdb1", "body.0", "conv_body", "conv_up1", "conv_up2")
BACKENDS = {"cpu": 1, "cuda": 2, "rocm": 3, "vulkan": 4}
PJRT = {"cpu": "cpu", "cuda": "iree_cuda", "rocm": "iree_rocm", "vulkan": "iree_vulkan"}
FP32_PTR = c.POINTER(c.c_float)


def reference(args):
    import torch

    profile = MODELS[args.model]
    scale = profile.get("scale", 4)
    engine = args.reference_engine or ("spandrel" if "runtime_model" in profile else "basicsr")
    if engine == "spandrel":
        import spandrel

        state, key = checkpoint_state(args.checkpoint, args.model)
        descriptor = spandrel.ModelLoader().load_from_state_dict(state)
        if descriptor.architecture.id != "ESRGAN" or descriptor.scale != scale:
            raise ValueError(f"Spandrel must identify an ESRGAN x{scale} model")
        model = descriptor.model
        taps = ("model.0", "model.1.sub.0.RDB1", "model.1.sub.0", f"model.1.sub.{profile['blocks']}",
                "model.4", "model.7")
        engine_version = {"spandrel_version": spandrel.__version__}
    else:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from basicsr.version import __gitsha__

        pin = PROVENANCE["reference"]["BasicSR"]["revision"]
        if len(__gitsha__) < 7 or not pin.startswith(__gitsha__):
            raise ValueError(f"BasicSR must be installed from Git at {pin}; found {__gitsha__}")
        state, key = read_checkpoint(args.checkpoint, args.model)
        model = RRDBNet(3, 3, scale=scale, num_feat=64, num_block=profile["blocks"], num_grow_ch=32)
        pytorch_shapes = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
        if pytorch_shapes != expected_shapes(args.model):
            raise AssertionError("Checkpoint schema differs from BasicSR RRDBNet")
        model.load_state_dict(state, strict=True)
        taps = TAPS
        engine_version = {"basic_sr_revision": pin}

    args.output.mkdir(parents=True, exist_ok=False)
    image_path = args.output / "input.png"
    if args.image is None:
        y, x = np.indices((9, 13))
        rgb = np.stack(((29 * x + 7 * y) % 256, (13 * y + 32 * (x % 2)) % 256, (x * y * 19) % 256), axis=-1)
        Image.fromarray(rgb.astype(np.uint8)).save(image_path)
    else:
        with Image.open(args.image) as image:
            if image.mode != "RGB" or image.format != "PNG":
                raise ValueError("--image must be a PNG in 8-bit RGB format")
            image.save(image_path)
    with Image.open(image_path) as image:
        data = np.asarray(image).transpose(2, 0, 1)[None].astype(np.float32) / np.float32(255.0)
        data = np.ascontiguousarray(data)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model = model.float().eval().to(args.torch_device)
    arrays = {"input": data}

    def hook(name):
        def capture(_model, _input, result):
            if engine == "basicsr" and name in ("conv_up1", "conv_up2"):
                result = torch.nn.functional.leaky_relu(result, negative_slope=0.2)
            arrays[name] = result.detach().cpu().clone().numpy()
        return capture

    handles = [model.get_submodule(path).register_forward_hook(hook(name)) for name, path in zip(TAPS, taps, strict=True)]
    try:
        with torch.inference_mode():
            value = torch.from_numpy(data).to(args.torch_device)
            if engine == "basicsr" and scale == 2:
                for axis, dimension in ((-1, value.shape[-1]), (-2, value.shape[-2])):
                    if dimension % 2:
                        pad = (0, 1, 0, 0) if axis == -1 else (0, 0, 0, 1)
                        value = torch.nn.functional.pad(value, pad, mode="reflect" if dimension > 1 else "replicate")
            result = model(value)
            arrays["output"] = result[:, :, :data.shape[2] * scale, :data.shape[3] * scale].cpu().numpy()
    finally:
        for handle in handles:
            handle.remove()
    for name, values in arrays.items():
        if values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError(f"{name}: expected finite FP32 values")
        np.save(args.output / f"{name}.npy", values, allow_pickle=False)
    rgb8 = np.rint(arrays["output"][0].clip(0, 1).transpose(1, 2, 0) * np.float32(255)).astype(np.uint8)
    Image.fromarray(rgb8).save(args.output / "expected.png")
    manifest = {
        "model": args.model,
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_key": key,
        "image_sha256": sha256(image_path),
        "precision": "float32",
        "torch_version": str(torch.__version__),
        "torch_device": args.torch_device,
        "reference_engine": engine,
        **engine_version,
    }
    (args.output / "reference.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"PyTorch reference: {args.output}")


def compare_tensor(name, actual, expected, atol, rtol):
    if actual.shape != expected.shape or actual.dtype != np.float32 or expected.dtype != np.float32:
        raise AssertionError(f"{name}: incompatible shape or dtype")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError(f"{name}: non-finite values")
    error = actual.astype(np.float64) - expected
    print(f"{name}: max_abs={np.max(np.abs(error)):.6g} rmse={np.sqrt(np.mean(error ** 2)):.6g}")
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=name)


def compare_cli(args, manifest, arrays, image_path):
    request = {
        "model": MODELS[manifest["model"]].get("runtime_model", manifest["model"]),
        "backend": args.backend,
        "weights_path": str(args.weights.resolve()),
        "image_path": str(image_path.resolve()),
        "output_dir": str(args.output.resolve()),
        "trace": True,
    }
    run = subprocess.run(
        [str(args.cli.resolve())], input=json.dumps(request).encode("utf-8"), stdout=subprocess.PIPE, check=False,
    )
    response = json.loads(run.stdout)
    height, width = arrays["output"].shape[2:]
    if (run.returncode or response.get("ok") is not True or response.get("backend") != args.backend
            or response.get("pjrt_platform") != PJRT[args.backend]
            or (response.get("height"), response.get("width")) != (height, width)):
        raise AssertionError(f"CLI failed or returned unexpected metadata: {response}")
    for name, expected in arrays.items():
        actual = np.fromfile(args.output / f"{name}.f32le", dtype="<f4").reshape(expected.shape)
        atol, rtol = (0, 0) if name == "input" else (args.atol, args.rtol)
        compare_tensor(name, actual, expected, atol, rtol)
    output_path = Path(response["output_path"]).resolve()
    scale = MODELS[manifest["model"]].get("scale", 4)
    if output_path != (args.output / f"realesrgan_x{scale}.png").resolve():
        raise AssertionError(f"Unexpected image path: {output_path}")
    with Image.open(output_path) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != (width, height):
            raise AssertionError(f"CLI output must be a PNG RGB8 at {scale}x input dimensions")
        actual_rgb8 = np.asarray(image)
    with Image.open(args.reference / "expected.png") as expected:
        delta = np.max(np.abs(actual_rgb8.astype(np.int16) - np.asarray(expected).astype(np.int16)))
    print(f"PNG: max_abs={delta} levels")
    if delta > args.max_rgb8_error:
        raise AssertionError(f"PNG differs by more than {args.max_rgb8_error} levels")


def compare_abi(args, variant, reference_input, expected):
    lib = c.CDLL(str(args.dll.resolve()))
    lib.zml_lib_session_open.argtypes = [c.c_uint32, c.POINTER(c.c_void_p)]
    lib.zml_lib_load_model.argtypes = [c.c_void_p, c.c_uint32, c.c_char_p]
    lib.zml_lib_infer.argtypes = [c.c_void_p, c.c_uint32, c.c_uint32, FP32_PTR, c.c_size_t, FP32_PTR, c.c_size_t]
    lib.zml_lib_session_open.restype = lib.zml_lib_load_model.restype = lib.zml_lib_infer.restype = c.c_uint32
    lib.zml_lib_session_close.argtypes = lib.zml_lib_unload_model.argtypes = [c.c_void_p]
    lib.zml_lib_session_close.restype = lib.zml_lib_unload_model.restype = None
    lib.zml_lib_last_error.argtypes = []
    lib.zml_lib_last_error.restype = c.c_char_p

    def check(status, expected_status=0):
        if status != expected_status:
            raise AssertionError(f"C ABI status {status}, expected {expected_status}: {lib.zml_lib_last_error()}")

    session = c.c_void_p()
    check(lib.zml_lib_session_open(BACKENDS[args.backend], c.byref(session)))
    path = str(args.weights.resolve()).encode("utf-8")
    scale = 2 if variant == RUNTIME_VARIANTS["RealESRGAN_x2plus"] else 4

    def infer(data, output_count=None):
        _, _, height, width = data.shape
        out = np.empty((1, 3, height * scale, width * scale), dtype=np.float32)
        status = lib.zml_lib_infer(
            session, width, height, data.ctypes.data_as(FP32_PTR), data.size,
            out.ctypes.data_as(FP32_PTR), out.size if output_count is None else output_count,
        )
        return status, out

    try:
        check(infer(reference_input)[0], 2)
        check(lib.zml_lib_load_model(session, variant, path))
        check(infer(reference_input, output_count=1)[0], 1)
        status, output = infer(reference_input)
        check(status)
        compare_tensor("C ABI first pass", output, expected, args.atol, args.rtol)
        status, other = infer(np.ascontiguousarray(1 - reference_input))
        check(status)
        if not np.isfinite(other).all() or np.allclose(other, output):
            raise AssertionError("C ABI failed to process a changed image")
        wrong_variant = 2 if variant == 1 else 1
        check(lib.zml_lib_load_model(session, wrong_variant, path), 4)
        status, output = infer(reference_input)
        check(status)
        compare_tensor("C ABI repeated pass after failed load", output, expected, args.atol, args.rtol)
        lib.zml_lib_unload_model(session)
        lib.zml_lib_unload_model(session)
        check(infer(reference_input)[0], 2)
        check(lib.zml_lib_load_model(session, variant, path))
        status, output = infer(reference_input)
        check(status)
        compare_tensor("C ABI pass after reload", output, expected, args.atol, args.rtol)
    finally:
        lib.zml_lib_session_close(session)
    print("C ABI lifecycle verified")


def compare(args):
    if not all(np.isfinite(v) and v >= 0 for v in (args.atol, args.rtol, args.max_rgb8_error)):
        raise ValueError("Tolerances must be finite and nonnegative")
    manifest = json.loads((args.reference / "reference.json").read_text(encoding="utf-8"))
    with safe_open(args.weights, framework="numpy", device="cpu") as weights:
        metadata = weights.metadata() or {}
        for key in ("model", "checkpoint_sha256", "checkpoint_key", "precision"):
            if metadata.get(key) != manifest.get(key):
                raise ValueError(f"Reference and SafeTensors metadata differ for {key}")
    image_path = args.reference / "input.png"
    if sha256(image_path) != manifest["image_sha256"]:
        raise ValueError("Reference image changed after PyTorch inference")
    arrays = {
        name: np.load(args.reference / f"{name}.npy", allow_pickle=False)
        for name in ("input", *TAPS, "output")
    }
    args.output.mkdir(parents=True, exist_ok=False)
    compare_cli(args, manifest, arrays, image_path)
    model = manifest["model"]
    compare_abi(args, RUNTIME_VARIANTS[MODELS[model].get("runtime_model", model)], arrays["input"], arrays["output"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    generate = commands.add_parser("reference")
    generate.add_argument("--model", choices=MODELS, required=True)
    generate.add_argument("--checkpoint", type=Path, required=True)
    generate.add_argument("--image", type=Path)
    generate.add_argument("--torch-device", default="cpu")
    generate.add_argument("--reference-engine", choices=("basicsr", "spandrel"))
    generate.add_argument("--output", type=Path, required=True)
    generate.set_defaults(run=reference)

    verify = commands.add_parser("compare")
    verify.add_argument("--reference", type=Path, required=True)
    verify.add_argument("--weights", type=Path, required=True)
    verify.add_argument("--cli", type=Path, required=True)
    verify.add_argument("--dll", type=Path, required=True)
    verify.add_argument("--backend", choices=BACKENDS, required=True)
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--atol", type=float, default=1e-3)
    verify.add_argument("--rtol", type=float, default=1e-3)
    verify.add_argument("--max-rgb8-error", type=int, default=1)
    verify.set_defaults(run=compare)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
