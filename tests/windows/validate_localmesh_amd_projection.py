"""Check the real DINOv3 patch projection on one Windows AMD PJRT backend."""

import argparse
from ctypes import POINTER, Structure, WinDLL, byref, c_size_t, c_ulong, c_ulonglong, sizeof
from datetime import datetime
import json
from math import fsum, prod
import os
from pathlib import Path
import struct
import subprocess
from tempfile import TemporaryDirectory
import time

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT.parent / "zml_win"
TOOLS = ENGINE / "bazel-win-tools"
GIB = 1024**3

GRAPH = """
module {
  func.func public @main(%image: tensor<1x3x32x32xf32>, %w: tensor<1024x3x16x16xf32>,
                         %bias: tensor<1024xf32>) -> tensor<1x1024x2x2xf32> {
    %patch = "stablehlo.convolution"(%image, %w) {
      dimension_numbers = #stablehlo.conv<[b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]>,
      batch_group_count = 1 : i64, feature_group_count = 1 : i64,
      window_strides = array<i64: 16, 16>, padding = dense<0> : tensor<2x2xi64>
    } : (tensor<1x3x32x32xf32>, tensor<1024x3x16x16xf32>) -> tensor<1x1024x2x2xf32>
    %b = stablehlo.broadcast_in_dim %bias, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x2x2xf32>
    %out = stablehlo.add %patch, %b : tensor<1x1024x2x2xf32>
    return %out : tensor<1x1024x2x2xf32>
  }
}
"""

MODEL = r"""
static void read_f32(const char *path, float *data, size_t count) {
    FILE *file;
    if (fopen_s(&file, path, "rb") != 0 || fread(data, sizeof(float), count, file) != count) {
        fprintf(stderr, "Could not read %s\n", path);
        exit(EXIT_FAILURE);
    }
    fclose(file);
}

static void check_model(PJRT_Client *client, PJRT_Device *device, bool expect_cpu, IDXGIAdapter3 *adapter) {
    char code[] = GRAPH_SOURCE;
    const int64_t image_dims[] = {1, 3, 32, 32}, weight_dims[] = {1024, 3, 16, 16}, bias_dims[] = {1024};
    float image[3 * 32 * 32], bias[1024], expected[4096], actual[4096];
    float *weights = malloc(1024 * 3 * 16 * 16 * sizeof(float));
    if (weights == NULL) exit(EXIT_FAILURE);
    read_f32("image.f32", image, 3 * 32 * 32);
    read_f32("weights.f32", weights, 1024 * 3 * 16 * 16);
    read_f32("bias.f32", bias, 1024);
    read_f32("expected.f32", expected, 4096);
    PJRT_LoadedExecutable *executable = compile(client, code);
    PJRT_Buffer *inputs[] = {
        upload(client, device, image, image_dims, 4),
        upload(client, device, weights, weight_dims, 4),
        upload(client, device, bias, bias_dims, 1),
    };
    free(weights);
    print_dxgi_usage(adapter, "With patch projection loaded");
    bool valid = true;
    for (int run = 0; run < 2; ++run) {
        const double begin = now_ms();
        execute(executable, device, expect_cpu, inputs, 3, actual, sizeof(actual));
        double max_abs = 0.0, sum_square = 0.0;
        for (size_t i = 0; i < 4096; ++i) {
            const double error = fabs((double)actual[i] - expected[i]);
            if (!isfinite(error) || error > 1e-4) valid = false;
            sum_square += error * error;
            if (error > max_abs) max_abs = error;
        }
        printf("DINOv3 patch run %d: 4096 values, max_abs=%.9g rmse=%.9g (%.3f ms)\n",
               run + 1, max_abs, sqrt(sum_square / 4096), now_ms() - begin);
    }
    CALL(PJRT_LoadedExecutable_Destroy, .executable = executable);
    for (size_t i = 0; i < 3; ++i) CALL(PJRT_Buffer_Destroy, .buffer = inputs[i]);
    if (!valid) exit(EXIT_FAILURE);
}
"""


def make_source():
    reference = (ENGINE / "tools/windows-pjrt-probe.c").read_text(encoding="utf-8")
    helpers, rest = reference.split("static void check_model(", 1)
    _, entry_point = rest.split("\nint wmain(", 1)
    return (
        "#include <math.h>\n" + helpers.replace("int64_t strides[2];", "int64_t strides[4];")
        + MODEL.replace("GRAPH_SOURCE", json.dumps(GRAPH)) + "\nint wmain(" + entry_point
    )


def make_inputs(checkpoint, case):
    with checkpoint.open("rb") as weights:
        header_size, = struct.unpack("<Q", weights.read(8))
        if not 0 < header_size <= 16 * 1024**2:
            raise ValueError("Invalid SafeTensors header size")
        header = json.loads(weights.read(header_size))
        for name, shape in (("weight", [1024, 3, 16, 16]), ("bias", [1024])):
            entry = header[f"embeddings.patch_embeddings.{name}"]
            first, last = entry["data_offsets"]
            size = 4 * prod(shape)
            if entry["dtype"] != "F32" or entry["shape"] != shape or first < 0 or last - first != size:
                raise ValueError(f"Unexpected DINOv3 patch projection {name}")
            weights.seek(8 + header_size + first)
            data = weights.read(size)
            if len(data) != size:
                raise ValueError(f"Truncated DINOv3 {name}")
            (case / ("weights.f32" if name == "weight" else "bias.f32")).write_bytes(data)
    pixels = [(index * 29 % 251 - 125) / 64 for index in range(3 * 32 * 32)]
    (case / "image.f32").write_bytes(struct.pack(f"<{len(pixels)}f", *pixels))
    patch_pixels = [
        [pixels[channel * 1024 + (y + row) * 32 + x + col]
         for channel in range(3) for row in range(16) for col in range(16)]
        for y in (0, 16) for x in (0, 16)
    ]
    bias = struct.unpack("<1024f", (case / "bias.f32").read_bytes())
    expected = []
    with (case / "weights.f32").open("rb") as weights:
        for offset in bias:
            kernel = struct.unpack("<768f", weights.read(768 * 4))
            expected.extend(fsum(p * w for p, w in zip(patch, kernel, strict=True)) + offset for patch in patch_pixels)
    (case / "expected.f32").write_bytes(struct.pack("<4096f", *expected))


class MemoryStatus(Structure):
    _fields_ = [("length", c_ulong), ("load", c_ulong)] + [
        (field, c_ulonglong)
        for field in ("phys", "free_phys", "page", "free_page", "virtual", "free_virtual", "reserved")
    ]


def limit_cpu(kernel):
    kernel.GetCurrentProcess.restype = c_size_t
    kernel.GetProcessAffinityMask.argtypes = [c_size_t, POINTER(c_size_t), POINTER(c_size_t)]
    kernel.SetProcessAffinityMask.argtypes = [c_size_t, c_size_t]
    kernel.SetPriorityClass.argtypes = [c_size_t, c_ulong]
    process = kernel.GetCurrentProcess()
    affinity, system = c_size_t(), c_size_t()
    if not kernel.GetProcessAffinityMask(process, byref(affinity), byref(system)):
        raise OSError("Cannot query CPU affinity")
    if not kernel.SetProcessAffinityMask(process, affinity.value & -affinity.value):
        raise OSError("Cannot restrict the test to one logical CPU")
    if not kernel.SetPriorityClass(process, subprocess.IDLE_PRIORITY_CLASS):
        raise OSError("Cannot set idle priority")


def available_memory(kernel):
    status = MemoryStatus(sizeof(MemoryStatus))
    if not kernel.GlobalMemoryStatusEx(byref(status)):
        raise OSError("Cannot query available physical memory")
    return status.free_phys


def run(command, env, case, log, kernel, timeout):
    if available_memory(kernel) < 6 * GIB:
        raise RuntimeError("Less than 6 GiB of physical memory available")
    deadline = time.monotonic() + timeout
    with subprocess.Popen(command, env=env, cwd=case, stdout=log, stderr=subprocess.STDOUT,
                          creationflags=subprocess.IDLE_PRIORITY_CLASS | subprocess.CREATE_NO_WINDOW) as child:
        try:
            while True:
                try:
                    return child.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    if time.monotonic() >= deadline or available_memory(kernel) < 6 * GIB:
                        raise RuntimeError("Time or system memory limit reached")
        finally:
            if child.poll() is None:
                log.write(b"Stopping the test process tree.\n")
                subprocess.run(["taskkill.exe", "/PID", str(child.pid), "/T", "/F"],
                               stdout=log, stderr=subprocess.STDOUT,
                               creationflags=subprocess.CREATE_NO_WINDOW, check=False)
                child.wait(timeout=10)


def test(args, case, log, kernel):
    options = {
        "IREE_PJRT_COMPILER_LIB_PATH": str(TOOLS / "iree-wheel/iree/compiler/_mlir_libs/IREECompiler.dll"),
        "CUDA_VISIBLE_DEVICES": "-1",
        "HIP_VISIBLE_DEVICES": "0" if args.backend == "rocm" else "-1",
        "OMP_NUM_THREADS": "1",
        "TEMP": str(case),
        "TMP": str(case),
        "TMPDIR": str(case),
    }
    env = os.environ.copy()
    for name in ("IREE_PJRT_IREE_COMPILER_OPTIONS", "IREE_PJRT_ROCM_TARGET", "VK_ICD_FILENAMES", "VK_DRIVER_FILES"):
        env.pop(name, None)
    if args.backend == "rocm":
        options["IREE_PJRT_ROCM_TARGET"] = "gfx1036"
    else:
        options["VK_DRIVER_FILES"] = str(args.amd_icd.resolve(strict=True))
        options["IREE_PJRT_IREE_COMPILER_OPTIONS"] = "--iree-vulkan-target=gfx1036"
    env.update(options)
    plugin = TOOLS / f"iree-build/python/iree/_pjrt_libs/{args.backend}/iree_pjrt_{args.backend}_dylib.dll"
    if not plugin.is_file():
        raise FileNotFoundError(plugin)
    make_inputs(args.checkpoint, case)
    source, exe = case / "projection.c", case / "projection.exe"
    source.write_text(make_source(), encoding="utf-8")
    log.write(f"{args.backend} gfx1036, 1 logical CPU, idle priority\nDLL {plugin}\n".encode())
    command = [args.clang, "-fuse-ld=lld", "/nologo", "/std:c11", f"/I{TOOLS}",
               f"/I{ENGINE / 'tools'}", source, ENGINE / "tools/windows-video-memory.c",
               f"/Fe{exe}", "dxgi.lib", "kernel32.lib"]
    return run(command, env, case, log, kernel, 25) or run(
        [exe, plugin, "gpu", "AMD Radeon"], env, case, log, kernel, 35
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("rocm", "vulkan"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--amd-icd", type=Path)
    parser.add_argument("--clang", default="clang-cl")
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("The probe requires native Windows")
    if args.backend == "vulkan" and not args.amd_icd:
        parser.error("Vulkan requires the AMD ICD manifest path")
    scratch = args.scratch.resolve(strict=True)
    if scratch.drive.upper() != "D:" or not scratch.is_dir():
        parser.error("Use the assigned scratch directory on D:")
    kernel = WinDLL("kernel32", use_last_error=True)
    limit_cpu(kernel)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = ROOT / "validation-output" / f"localmesh-{args.backend}-dinov3-patches-{stamp}.log"
    output.parent.mkdir(exist_ok=True)
    try:
        with TemporaryDirectory(prefix=f"localmesh-{args.backend}-", dir=scratch) as directory:
            with output.open("xb", buffering=0) as log:
                status = test(args, Path(directory), log, kernel)
    finally:
        print(output, flush=True)
    text = output.read_text(encoding="utf-8", errors="replace")
    print(text, end="")
    if not status and f"platform iree_{args.backend}, device 0: AMD Radeon" not in text:
        raise RuntimeError("Unexpected PJRT platform or device")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
