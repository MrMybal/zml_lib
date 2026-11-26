"""Run the Qwen Image Edit 2511 comparisons on a remote Linux x64 NVIDIA L4."""

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
IMAGE_TARGET = "//libraries/qwen_image:qwen_image_probe"
ENCODER_TARGET = "//libraries/qwen25_vl:qwen25_vl_encode"
PREPARED_DENOISER_BYTES = 20_738_759_084
LINUX_CUDA_FLAGS = (
    "--platforms=@zml//platforms:linux_amd64",
    "--@zml//platforms:cuda=true",
    "--@zml//platforms:cpu=false",
    "--@zml//platforms:rocm=false",
    "--@zml//platforms:vulkan=false",
    "--http_timeout_scaling=3.0",
    "--repo_contents_cache=",
)


def query_l4(uuid):
    command = [
        "nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    if uuid:
        command.extend(("--id", uuid))
    rows = list(csv.reader(subprocess.check_output(command, text=True).splitlines(), skipinitialspace=True))
    if len(rows) != 1:
        raise ValueError(f"Expected one allocated GPU, found {len(rows)}. Select it with --gpu-uuid.")
    gpu, name, driver, total, free = rows[0]
    if not gpu.startswith("GPU-") or "L4" not in name.split():
        raise ValueError(f"Expected an NVIDIA L4, found {name}")
    print(json.dumps({"gpu": name, "uuid": gpu, "driver": driver,
                      "vram_mib": int(total), "free_vram_mib": int(free)}), flush=True)
    return gpu, int(free)


def linux_executable(path):
    path = path.resolve(strict=True)
    with path.open("rb") as binary:
        header = binary.read(20)
    if (header[:6] != b"\x7fELF\x02\x01" or header[18:20] != b"\x3e\x00"
            or not os.access(path, os.X_OK)):
        raise ValueError(f"Expected an executable Linux x64 ELF binary at {path}")
    return path


def build_reference_cuda_math(probe, nvcc, scratch):
    probe = linux_executable(probe)
    runfiles = probe.with_name(probe.name + ".runfiles")
    packages = list(runfiles.glob("*/sandbox/lib/libzml_cuda.so"))
    if len(packages) != 1:
        raise ValueError(f"Expected one bundled CUDA PJRT library in {runfiles}")
    cuda_lib = packages[0].parent
    for name in ("libcublasLt.so.13", "libcublas.so.13"):
        if not (cuda_lib / name).is_file():
            raise ValueError(f"Missing {name} in the CUDA runfiles")
    output = scratch / "libzml_qwen_cuda_math.so"
    source = Path(__file__).with_name("cuda_math.cu")
    temp = scratch / "cuda-math-tmp"
    temp.mkdir()
    env = dict(os.environ, TEMP=str(temp), TMP=str(temp), TMPDIR=str(temp))
    subprocess.run([
        nvcc, "-O3", "-arch=sm_89", "-shared", "-Xcompiler=-fPIC,-fvisibility=hidden",
        str(source), "-L" + str(cuda_lib), "-Xlinker", "-l:libcublasLt.so.13",
        "-Xlinker", "-l:libcublas.so.13", "-Xlinker", "-rpath",
        "-Xlinker", str(cuda_lib), "-o", str(output),
    ], env=env, check=True)
    return output


def validation_command(args, weights, image_probe, encoder_probe):
    image_probe = linux_executable(image_probe)
    if args.cuda_low_memory:
        capability = subprocess.run(
            [image_probe, "--cuda-low-memory", "edit_2511", "schema"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        if capability.returncode:
            raise ValueError("Use a Qwen image probe built with --cuda-low-memory support")
    command = [
        sys.executable, "-B", "-u", str(Path(__file__).with_name("validate.py")),
        "--probe", str(image_probe), "--comfy", str(args.comfy),
        "--backend", "cuda", "--expected-pjrt", "cuda", "--expected-device", "cuda:0",
        "--vae", str(weights["vae"]), "--denoiser", str(weights["denoiser"]),
        "--encoder", str(weights["encoder"]), "--encoder-probe", str(linux_executable(encoder_probe)),
        "--tokenizer", str((args.comfy / "comfy/text_encoders/qwen25_tokenizer").resolve(strict=True)),
        "--reference-device", "cuda:0", "--feedback-endpoints",
        "--width", str(args.width), "--height", str(args.height),
    ]
    if args.vision_size is not None:
        command.extend(("--vision-size", str(args.vision_size)))
    if args.cuda_qk:
        command.append("--cuda-qk")
    if args.cuda_low_memory:
        command.append("--cuda-low-memory")
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True, help="temporary directory on the VM's local disk")
    parser.add_argument("--comfy", type=Path, required=True, help="ComfyUI source checkout")
    parser.add_argument("--models", type=Path, required=True, help="models directory with the ComfyUI layout")
    parser.add_argument("--width", type=int, default=64, help="reference image width")
    parser.add_argument("--height", type=int, default=48, help="reference image height")
    parser.add_argument("--vision-size", type=int, help="square conditioning image size for Qwen2.5-VL")
    parser.add_argument("--cuda-qk", action="store_true", help="use the ordered Q/K graph")
    parser.add_argument("--cuda-low-memory", action="store_true",
                        help="run the denoiser block by block with the CUDA BFC allocator")
    parser.add_argument("--cuda-reference-math", action="store_true",
                        help="build the CUDA arithmetic used for the Edit 2511 parity check on L4")
    parser.add_argument("--nvcc", default="nvcc", help="CUDA compiler for --cuda-reference-math")
    parser.add_argument("--gpu-uuid", help="allocated L4 UUID when nvidia-smi lists several GPUs")
    probes = parser.add_mutually_exclusive_group(required=True)
    probes.add_argument("--bazel", type=Path, help="Bazel 9.3.0 or Bazelisk. Build probes in scratch.")
    probes.add_argument("--probe", type=Path, help="prebuilt Linux x64 qwen_image_probe with its runfiles")
    parser.add_argument("--encoder-probe", type=Path, help="prebuilt Linux x64 qwen25_vl_encode with its runfiles")
    args = parser.parse_args()
    if sys.platform != "linux" or platform.machine() != "x86_64":
        parser.error("Run this on the remote Linux x86_64 worker")
    if args.probe and not args.encoder_probe:
        parser.error("--probe requires --encoder-probe")
    if args.encoder_probe and not args.probe:
        parser.error("--encoder-probe is only needed with --probe")
    if args.cuda_reference_math and (not args.cuda_low_memory or args.cuda_qk):
        parser.error("--cuda-reference-math requires the standard --cuda-low-memory profile")
    args.comfy = args.comfy.resolve(strict=True)
    if not (args.comfy / "comfy/options.py").is_file():
        parser.error(f"ComfyUI sources were not found under {args.comfy}")
    weights = {
        "vae": args.models / "vae/qwen_image_vae.safetensors",
        "denoiser": args.models / "diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors",
        "encoder": args.models / "text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
    }
    for name, path in weights.items():
        if not path.is_file():
            parser.error(f"Missing {name} checkpoint: {path}")
        weights[name] = path.resolve(strict=True)
    if not args.scratch.is_absolute():
        parser.error("--scratch must be an absolute path on the worker")
    gpu, _ = query_l4(args.gpu_uuid)
    print(f"Prepared denoiser checkpoint: {PREPARED_DENOISER_BYTES / 2**30:.2f} GiB on disk", flush=True)

    os.environ["ZML_TEMP_ROOT"] = str(args.scratch)
    sys.path.insert(0, str(ROOT))
    from scripts import build as builder

    with builder.scratch_directory() as scratch:
        if args.bazel:
            bazel = args.bazel.resolve(strict=True)
            env = builder.run_environment(scratch, bazel.parent)
            env["BAZEL_SH"] = shutil.which("bash") or "/bin/bash"
            revision = builder.check_engine(ROOT.parent / "zml_win", env)
            print(f"ZML engine revision: {revision}", flush=True)
            binaries = builder.build(str(bazel), ROOT, scratch, env, "qwen-linux", LINUX_CUDA_FLAGS,
                                     (IMAGE_TARGET, ENCODER_TARGET))
            args.probe = binaries / "libraries/qwen_image/qwen_image_probe"
            args.encoder_probe = binaries / "libraries/qwen25_vl/qwen25_vl_encode"
        command = validation_command(args, weights, args.probe, args.encoder_probe)
        env = dict(os.environ, ZML_TEMP_ROOT=str(scratch), CUDA_VISIBLE_DEVICES=gpu,
                   NVIDIA_TF32_OVERRIDE="0", PYTHONDONTWRITEBYTECODE="1")
        if args.cuda_reference_math:
            env["ZML_QWEN_CUDA_MATH"] = str(build_reference_cuda_math(args.probe, args.nvcc, scratch))
            env.setdefault("ZML_QWEN_DENOISER_XLA_FLAGS",
                           "--xla_gpu_enable_triton_gemm=false --xla_gpu_enable_cublaslt=false")
        subprocess.run(command, env=env, cwd=ROOT, check=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from None
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from None
