"""Validate the CyTools restoration checkpoints against their ZML ports."""

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import spandrel
import torch
from safetensors.torch import save_file

from build import ROOT, TEMP_ROOT, scratch_directory
from probe_placement import PLATFORMS, verify_placement


RESTORATION = Path("CyTools_VideoStudio/runtime/restoration-validation")
X2_WEIGHTS = Path("CyTools_VideoToStereo3D/data/models/upscale-realesrgan-x2plus/RealESRGAN_x2plus.pth")
PROFILES = {
    "srvggnet": (RESTORATION / "realesr-general-x4v3.pth", 4, 101),
    "swinir": (RESTORATION / "006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg10.pth", 1, 490),
    "swinir-jpeg20": (RESTORATION / "006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg20.pth", 1, 490),
    "swinir-colordn25": (RESTORATION / "005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth", 1, 490),
}
SWINIR_PROFILES = {
    "swinir": (7, 255, "swinir_probe"),
    "swinir-jpeg20": (7, 255, "swinir_probe"),
    "swinir-colordn25": (8, 1, "swinir_color_dn_probe"),
}


def input_image(height, width):
    y, x = np.indices((height, width))
    planes = ((29 * x + 7 * y) % 256, (13 * y + 32 * (x % 2)) % 256, (x * y * 19) % 256)
    return np.ascontiguousarray(np.stack(planes)[None].astype(np.float32) / 255)


def run(arguments):
    subprocess.run([str(value) for value in arguments], cwd=ROOT, check=True)


def check_simple(profile, checkpoint, work, bins, backend, expected_device):
    _, scale, count = PROFILES[profile]
    descriptor = spandrel.ModelLoader().load_from_file(str(checkpoint))
    if descriptor.scale != scale:
        raise ValueError(f"{profile}: expected scale {scale}, got {descriptor.scale}")
    model = descriptor.model.cpu().float().eval()
    library, target = profile, f"{profile}_probe"
    dimensions = ((9, 13), (14, 21))
    if profile in SWINIR_PROFILES:
        window, image_range, target = SWINIR_PROFILES[profile]
        library = "swinir"
        if model.window_size != window or model.img_range != image_range:
            raise ValueError(f"{profile}: expected a {window}×{window} window and image range {image_range}")
        if window == 8:
            dimensions = ((9, 13), (16, 24))
    parameters = {key: value.detach().contiguous() for key, value in model.named_parameters()}
    if len(parameters) != count:
        raise ValueError(f"{profile}: expected {count} learned parameters, got {len(parameters)}")
    with checkpoint.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    print(f"{profile} checkpoint={digest} tensors={count}", flush=True)
    weights = work / "weights.safetensors"
    if bins:
        save_file(parameters, weights, metadata={"source_sha256": digest, "precision": "float32"})
    for height, width in dimensions:
        image = input_image(height, width)
        with torch.inference_mode():
            expected = model(torch.from_numpy(image)).numpy()
        shape = (1, 3, height * scale, width * scale)
        if expected.shape != shape or not np.isfinite(expected).all():
            raise ValueError(f"{profile}: invalid reference output")
        if not bins:
            print(f"{profile} PyTorch {height}x{width} -> {expected.shape}", flush=True)
            continue
        source = work / f"{height}x{width}-input.f32le"
        result = work / f"{height}x{width}-output.f32le"
        image.astype("<f4", copy=False).tofile(source)
        probe = bins / "libraries" / library / f"{target}.exe"
        process = subprocess.run(
            [str(probe), backend, str(weights), str(height), str(width), str(source), str(result)],
            cwd=ROOT, capture_output=True, text=True, errors="replace",
        )
        print(process.stderr, end="", file=sys.stderr)
        if process.returncode:
            print(f"{profile} exited with status {process.returncode:#x}", file=sys.stderr)
        process.check_returncode()
        verify_placement(process.stderr, backend, 2, expected_device)
        actual = np.fromfile(result, dtype="<f4").reshape(shape)
        np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3, equal_nan=False)
        print(f"{profile} {backend} {height}x{width} max_abs={np.max(np.abs(actual - expected)):.9g}", flush=True)


def check_rrdb(checkpoint, work, bins, backend):
    scripts = ROOT / "libraries/realesrgan/scripts"
    weights = work / "weights.safetensors"
    reference = work / "reference"
    variant = "RealESRGAN_x2plus"
    run([sys.executable, "-B", scripts / "validate.py", "reference", "--reference-engine", "spandrel",
         "--model", variant, "--checkpoint", checkpoint, "--output", reference])
    if not bins:
        return
    run([sys.executable, "-B", scripts / "convert.py", "--model", variant, "--checkpoint", checkpoint, "--output", weights])
    run([sys.executable, "-B", scripts / "validate.py", "compare", "--reference", reference, "--weights", weights,
         "--cli", bins / "libraries/realesrgan/realesrgan_cli.exe", "--dll", bins / "zml_lib.dll",
         "--backend", backend, "--output", work / "zml"])
    run([bins / "zml_lib_cpp.exe", weights, "x2plus", backend])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cytools", type=Path, required=True)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--profile", action="append", choices=(*PROFILES, "realesrgan-x2plus"))
    parser.add_argument("--checkpoint", type=Path, help="override the checkpoint for a single selected profile")
    parser.add_argument("--backend", choices=PLATFORMS, default="cpu")
    parser.add_argument("--expected-device")
    args = parser.parse_args()
    selected = args.profile or ["srvggnet", "swinir", "realesrgan-x2plus"]
    if args.checkpoint and len(selected) != 1:
        parser.error("--checkpoint requires exactly one --profile")
    bins = None
    if not args.reference_only:
        path = os.environ.get("ZML_BIN_DIR")
        if not path:
            parser.error("Run Windows validation through scripts/build.py --then")
        bins = Path(path).resolve(strict=True)
        if not bins.is_relative_to(TEMP_ROOT):
            parser.error("Expected compiled binaries under _Temp")
    torch.set_num_threads(4)
    for profile in selected:
        relative = X2_WEIGHTS if profile == "realesrgan-x2plus" else PROFILES[profile][0]
        checkpoint = (args.checkpoint or args.cytools / relative).resolve(strict=True)
        with scratch_directory() as work:
            if profile == "realesrgan-x2plus":
                check_rrdb(checkpoint, work, bins, args.backend)
            else:
                check_simple(profile, checkpoint, work, bins, args.backend, args.expected_device)


if __name__ == "__main__":
    main()
