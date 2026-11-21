"""Check Real-ESRGAN CLI and ABI using deterministic local weights with a known output."""

import argparse
import ctypes as c
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "libraries/realesrgan/scripts"))
from checkpoints import MODELS, RUNTIME_VARIANTS, expected_shapes  # noqa: E402

BACKENDS = {"cpu": (1, "cpu"), "cuda": (2, "iree_cuda"), "rocm": (3, "iree_rocm"), "vulkan": (4, "iree_vulkan")}
FP32 = c.POINTER(c.c_float)


def fixture(path, variant):
    tensors = {name: np.zeros(shape, np.float32) for name, shape in expected_shapes(variant).items()}
    residual = (1 + 0.2 * 1.05 ** 3) * (1.2 ** (MODELS[variant]["blocks"] - 1))
    for channel in range(3):
        for layer in ("conv_first", "conv_up1", "conv_up2", "conv_hr"):
            tensors[f"{layer}.weight"][channel, channel, 1, 1] = 1
        for dense in range(1, 4):
            tensors[f"body.0.rdb{dense}.conv5.weight"][channel, channel, 1, 1] = 0.25
        tensors["conv_body.weight"][channel, channel, 1, 1] = 1 / residual
        tensors["conv_last.weight"][channel, channel, 1, 1] = 0.5
    save_file(tensors, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    args = parser.parse_args()
    binary = args.bin.resolve(strict=True)
    cli = binary / "realesrgan_cli.exe"
    if not cli.is_file():
        cli = binary / "libraries/realesrgan/realesrgan_cli.exe"
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    backend, pjrt = BACKENDS[args.backend]
    lib = c.CDLL(str(binary / "zml_lib.dll"))
    lib.zml_lib_session_open.argtypes = [c.c_uint32, c.POINTER(c.c_void_p)]
    lib.zml_lib_load_model.argtypes = [c.c_void_p, c.c_uint32, c.c_char_p]
    lib.zml_lib_infer.argtypes = [c.c_void_p, c.c_uint32, c.c_uint32, FP32, c.c_size_t, FP32, c.c_size_t]
    lib.zml_lib_session_close.argtypes = [c.c_void_p]
    lib.zml_lib_session_close.restype = None
    lib.zml_lib_last_error.restype = c.c_char_p

    def check(status):
        if status:
            raise RuntimeError(f"ABI status {status}: {lib.zml_lib_last_error()}")

    rgb = (np.arange(27, dtype=np.uint8).reshape(3, 3, 3) * 7 + 3)
    image = out / "input.png"
    Image.fromarray(rgb).save(image)
    source = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / np.float32(255)
    reference = source.repeat(4, axis=1).repeat(4, axis=2)
    session = c.c_void_p()
    check(lib.zml_lib_session_open(backend, c.byref(session)))
    try:
        for variant, variant_id in RUNTIME_VARIANTS.items():
            weights = out / f"{variant}.safetensors"
            fixture(weights, variant)
            argv = {
                "model": variant, "weights_path": str(weights), "image_path": str(image),
                "output_dir": str(out / variant), "backend": args.backend, "trace": True,
            }
            result = subprocess.run([cli], input=json.dumps(argv).encode(),
                                    stdout=subprocess.PIPE, check=True)
            response = json.loads(result.stdout)
            if response["pjrt_platform"] != pjrt or response["backend"] != args.backend:
                raise AssertionError(response)
            raw = np.fromfile(out / variant / "output.f32le", dtype="<f4").reshape(reference.shape)
            np.testing.assert_allclose(raw, reference, atol=4e-5, rtol=0)
            with Image.open(response["output_path"]) as png:
                np.testing.assert_array_equal(np.asarray(png), rgb.repeat(4, axis=0).repeat(4, axis=1))
            check(lib.zml_lib_load_model(session, variant_id, str(weights).encode("utf-8")))
            actual = np.empty_like(reference)
            check(lib.zml_lib_infer(session, 3, 3, source.ctypes.data_as(FP32), source.size,
                                    actual.ctypes.data_as(FP32), actual.size))
            np.testing.assert_allclose(actual, reference, atol=4e-5, rtol=0)
            print(f"{args.backend} / {pjrt} {variant}: CLI and C ABI OK", flush=True)
    finally:
        lib.zml_lib_session_close(session)


if __name__ == "__main__":
    main()
