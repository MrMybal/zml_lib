"""Run the draft LocalMesh workflow with real weights on Windows without NVIDIA."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
BACKENDS = ("cpu", "rocm", "vulkan")
PROBES = {
    "birefnet": "birefnet_probe",
    "dinov3": "dinov3_condition",
    "trellis": "trellis_structure",
    "trellis2": "trellis2_flow",
    "trellis2_vae": "trellis2_vae_probe",
}


def checked_file(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def prepare_birefnet(python, scratch, probe):
    weight = scratch / "converted" / "birefnet.safetensors"
    if not weight.is_file():
        weight.parent.mkdir(exist_ok=True)
        subprocess.run(
            [python, "-B", ROOT / "libraries/birefnet/prepare.py", "--probe", probe,
             "--checkpoint", checked_file(scratch / "birefnet/model.safetensors"), "--output", weight],
            check=True,
        )
    return weight


def run_prepare(python, scratch, bins):
    models = scratch / "models"
    original = models / "TRELLIS.2-4B" / "ckpts_fp8"
    converted = scratch / "converted"
    converted.mkdir(exist_ok=True)

    def command(script, executable, *options):
        subprocess.run(
            [python, "-B", ROOT / "libraries" / script, "--probe", executable, *options],
            check=True,
        )

    stages = {}
    weight = prepare_birefnet(python, scratch, bins["birefnet"])
    stages["birefnet_hr"] = (bins["birefnet"], weight)
    stages["dinov3"] = (bins["dinov3"], checked_file(scratch / "dinov3" / "model.safetensors"))
    weight = converted / "structure.safetensors"
    if not weight.is_file():
        original_structure = checked_file(
            models / "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors"
        )
        subprocess.run(
            [python, "-B", ROOT / "libraries/trellis/structure_io.py", "convert",
             "--probe", bins["trellis"], "--checkpoint", original_structure, "--weights", weight],
            check=True,
        )
    stages["structure"] = (bins["trellis"], weight)
    for stage, profile, name in (
        ("structure_flow", "structure", "ss_flow_img_dit_1_3B_64_fp8"),
        ("shape_512", "shape", "slat_flow_img2shape_dit_1_3B_512_fp8"),
        ("texture_512", "texture", "slat_flow_imgshape2tex_dit_1_3B_512_fp8"),
    ):
        weight = converted / f"{stage}.safetensors"
        if not weight.is_file():
            command("trellis2/prepare.py", bins["trellis2"], "--profile", profile,
                    "--config", checked_file(original / f"{name}.json"),
                    "--checkpoint", checked_file(original / f"{name}.safetensors"),
                    "--output", weight)
        stages[stage] = (bins["trellis2"], weight)
    for stage, name in (
        ("shape_decode", "shape_dec_next_dc_f16c32_fp8"),
        ("texture_decode", "tex_dec_next_dc_f16c32_fp8"),
    ):
        weight = converted / stage
        if not (weight / "manifest.json").is_file():
            command("trellis2_vae/prepare.py", bins["trellis2_vae"], "--profile", stage,
                    "--config", checked_file(original / f"{name}.json"),
                    "--checkpoint", checked_file(original / f"{name}.safetensors"),
                    "--output", weight)
        stages[stage] = (bins["trellis2_vae"], weight)
    return stages


def select_backend(backend, tools, amd_icd):
    options = {
        "CUDA_VISIBLE_DEVICES": "-1",
        "HIP_VISIBLE_DEVICES": "-1",
        "ZML_CPU_PJRT_DLL": str(checked_file(tools / "pjrt-cpu" / "pjrt_cpu.dll")),
    }
    if backend == "cpu":
        return "cpu", "cpu:0", options
    library = checked_file(
        tools / f"iree-build/python/iree/_pjrt_libs/{backend}/iree_pjrt_{backend}_dylib.dll"
    )
    options[f"ZML_{backend.upper()}_PJRT_DLL"] = str(library)
    options["IREE_PJRT_COMPILER_LIB_PATH"] = str(checked_file(
        tools / "iree-wheel/iree/compiler/_mlir_libs/IREECompiler.dll"
    ))
    if backend == "rocm":
        options.update(HIP_VISIBLE_DEVICES="0", IREE_PJRT_ROCM_TARGET="gfx1036")
    else:
        options["VK_DRIVER_FILES"] = str(checked_file(amd_icd))
        options["IREE_PJRT_IREE_COMPILER_OPTIONS"] = "--iree-vulkan-target=gfx1036"
    return f"iree_{backend}", "AMD Radeon", options


def case_environment(backend, tools, amd_icd, case, scratch):
    pjrt, device, options = select_backend(backend, tools, amd_icd)
    env = os.environ.copy()
    for name in ("IREE_PJRT_IREE_COMPILER_OPTIONS", "IREE_PJRT_ROCM_TARGET", "ZML_CUDA_PJRT_DLL",
                 "ZML_ROCM_PJRT_DLL", "ZML_VULKAN_PJRT_DLL", "VK_DRIVER_FILES",
                 "VK_ICD_FILENAMES", "VK_ADD_DRIVER_FILES", "XLA_FLAGS", "HIP_VISIBLE_DEVICES"):
        env.pop(name, None)
    env.update(options, LOCALMESH_ROOT=str(scratch), TMPDIR=str(case), TEMP=str(case),
               TMP=str(case), ZML_TEMP_ROOT=str(case), HF_HUB_OFFLINE="1",
               PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", LUMENGEN_SANS_REPLI="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return pjrt, device, env


def guarded_run(command, env, log, timeout=3 * 3600):
    from ctypes import WinDLL
    from validate_localmesh_amd_projection import run

    with log.open("wb", buffering=0) as stream:
        return run(command, env, ROOT, stream, WinDLL("kernel32", use_last_error=True), timeout)


def validate_matte(backend, probe, weight, scratch, photo, tools, amd_icd, tag, size, raw):
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(ROOT))
    from scripts.probe_placement import verify_placement

    output_dir = ROOT / "validation-output"
    output_dir.mkdir(exist_ok=True)
    mode = "logits" if raw else "alpha"
    with tempfile.TemporaryDirectory(prefix=f"matte-{backend}-", dir=scratch) as directory:
        case = Path(directory)
        src = case / "image.f32"
        with Image.open(photo) as image:
            rgb = np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR),
                             dtype=np.float32) / 255
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        ((rgb - mean) / std).transpose(2, 0, 1).astype("<f4", copy=False).tofile(src)

        def read_result(target):
            pjrt, device, env = case_environment(target, tools, amd_icd, case, scratch)
            log = output_dir / f"localmesh-birefnet-{backend}-{target}-{size}-{mode}-{tag}.log"
            dest = case / f"{target}.f32"
            args = [probe, target, weight, str(size), str(size), src, dest, "1"]
            if raw:
                args.append("--logits")
            print(f"BiRefNet {target} {size}x{size} {mode}. Log: {log}", flush=True)
            if code := guarded_run(args, env, log, timeout=600):
                raise RuntimeError(f"BiRefNet {target} exited with {code}. See {log}")
            verify_placement(log.read_text(encoding="utf-8", errors="replace"), target, 1, device,
                             expected_pjrt=pjrt)
            values = np.fromfile(dest, dtype="<f4")
            if len(values) != size * size or not np.isfinite(values).all():
                raise ValueError(f"Invalid BiRefNet {target} output")
            return values

        reference = read_result("cpu")
        result = read_result(backend) if backend != "cpu" else reference
        rmse = float(np.sqrt(np.mean((result.astype(np.float64) - reference)**2)))
        cpu_fg = int(np.count_nonzero(reference >= (np.log(4) if raw else .8)))
        gpu_fg = int(np.count_nonzero(result >= (np.log(4) if raw else .8)))
        report = dict(backend=backend, size=size, output=mode, rmse=rmse,
                      cpu_foreground=cpu_fg, backend_foreground=gpu_fg,
                      minimum=float(result.min()), maximum=float(result.max()))
        print(json.dumps(report), flush=True)
        (output_dir / f"localmesh-birefnet-{backend}-{size}-{mode}-{tag}.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8",
        )
        return rmse <= .001 and bool(gpu_fg) == bool(cpu_fg) and (size != 1024 or cpu_fg > 0)


def validate_backend(backend, stages, python, scratch, fork, tools, amd_icd, tag):
    with tempfile.TemporaryDirectory(prefix=f"{backend}-", dir=scratch) as directory:
        case = Path(directory)
        config = case / "config.json"
        log = case / "run.log"
        report = case / "report.json"
        output = case / "output"
        photo = case / "samurai.png"
        pjrt = device = None
        error = None
        try:
            pjrt, device, env = case_environment(backend, tools, amd_icd, case, scratch)
            subprocess.run(
                [python, "-B", "-c", "from PIL import Image; import sys; "
                 "Image.open(sys.argv[1]).convert('RGB').save(sys.argv[2])",
                 fork / "examples/samurai/front.png", photo], env=env, check=True,
            )
            config.write_text(json.dumps({
                "backend": backend, "pjrt": pjrt, "device": device, "scratch": str(case),
                "stages": {key: {"probe": str(probe), "weights": str(weights)}
                           for key, (probe, weights) in stages.items()},
            }), encoding="utf-8")
            argv = [python, "-u", "-B", fork / "tests/validate_zml_localmesh.py",
                    "--config", config, "--photo", photo, "--tier", "draft", "--seed", "42",
                    "--topology", "quad", "--output", output, "--report", report]
            for name in stages:
                argv += ["--require-stage", name]
            if status := guarded_run(argv, env, log):
                error = f"LocalMesh exited with status {status}"
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            error = str(exc)
        result = {
            "backend": backend, "expected_pjrt": pjrt, "expected_device": device,
            "error": error, "result": json.loads(report.read_text()) if report.is_file() else None,
        }
        (case / "execution.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        archive = ROOT / "validation-output" / f"localmesh-windows-quad-{backend}-{tag}.tar.gz"
        partial = archive.with_suffix(".part")
        try:
            with tarfile.open(partial, "x:gz") as stream:
                for artifact in (log, report, config, case / "execution.json", output):
                    if artifact.exists():
                        stream.add(artifact, arcname=artifact.name)
            partial.replace(archive)
        finally:
            partial.unlink(missing_ok=True)
        print(json.dumps({"archive": str(archive), **result}), flush=True)
        return error is None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--backend", choices=BACKENDS, action="append")
    parser.add_argument("--amd-icd", type=Path)
    parser.add_argument("--matte-size", type=int, help="Run only the BiRefNet matte against ZML CPU")
    parser.add_argument("--logits", action="store_true", help="Compare BiRefNet outputs before sigmoid")
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("Use the LocalMesh validation driver directly on Linux")
    if args.matte_size is not None and (not 64 <= args.matte_size <= 1024 or args.matte_size % 64):
        parser.error("--matte-size must be divisible by 64 and between 64 and 1024")
    if args.logits and args.matte_size is None:
        parser.error("--logits requires --matte-size")
    from ctypes import WinDLL
    from validate_localmesh_amd_projection import limit_cpu

    limit_cpu(WinDLL("kernel32", use_last_error=True))
    scratch = (args.assets or Path(os.environ["ZML_TEMP_ROOT"]) / "localmesh-native").resolve(strict=True)
    if scratch.drive.upper() == "C:":
        parser.error("Use a temporary directory outside C:")
    fork = ROOT.parent / "localmesh-engine_zml"
    tools = ROOT.parent / "zml_win" / "bazel-win-tools"
    python = sys.executable if args.matte_size else checked_file(scratch / "venv/Scripts/python.exe")
    bins_dir = scratch / "bin"
    bins_dir.mkdir(exist_ok=True)
    bins = {}
    required = {"birefnet": PROBES["birefnet"]} if args.matte_size else PROBES
    for library, basename in required.items():
        name = f"{basename}.exe"
        destination = bins_dir / name
        if "ZML_BIN_DIR" in os.environ:
            shutil.copy2(checked_file(Path(os.environ["ZML_BIN_DIR"]) / "libraries" / library / name),
                         destination)
        bins[library] = checked_file(destination)
    matches = tuple(Path(os.environ["SystemRoot"]).glob(
        "System32/DriverStore/FileRepository/u0420529.inf_amd64_*/*/amd-vulkan64.json"
    ))
    icd = args.amd_icd or (matches[0] if len(matches) == 1 else Path("AMD-ICD-not-selected"))
    cases = args.backend or BACKENDS
    tag = time.strftime("%Y%m%d-%H%M%S")
    if args.matte_size:
        probe = bins["birefnet"]
        weight = prepare_birefnet(python, scratch, probe)
        photo = fork / "examples/samurai/front.png"
        passed = [validate_matte(backend, probe, weight, scratch, photo, tools, icd,
                                 tag, args.matte_size, args.logits) for backend in cases]
        return 0 if all(passed) else 1
    stages = run_prepare(python, scratch, bins)
    passed = [validate_backend(backend, stages, python, scratch, fork, tools, icd, tag)
              for backend in cases]
    return 0 if all(passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
