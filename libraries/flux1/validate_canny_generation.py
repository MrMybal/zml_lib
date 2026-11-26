"""Compare the direct FLUX.1 Canny Dev launcher and the ImageStudio CLI on CUDA."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from PIL import Image, ImageDraw

from validate_generation import HERE, image_size, local_environment, measure, scratch_directory, verify_placement


def control_image(path, width, height):
    image = Image.new("RGB", (width, height), (36, 47, 63))
    draw = ImageDraw.Draw(image)
    draw.ellipse((width // 4, height // 8, width * 3 // 4, height * 5 // 8), fill=(215, 194, 119))
    draw.polygon([(0, height - 1), (width // 2, height // 3), (width - 1, height - 1)], fill=(28, 113, 152))
    draw.rectangle((width // 6, height * 3 // 4, width * 5 // 6, height * 7 // 8), fill=(243, 232, 211))
    image.save(path)


def inspect_png(path, dimensions):
    with Image.open(path) as png:
        if (png.format, png.mode, png.size) != ("PNG", "RGB", dimensions):
            raise AssertionError("Expected a PNG RGB with the requested dimensions")
        if all(low == high for low, high in png.getextrema()):
            raise AssertionError("Canny Dev produced a constant image")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate(args, work):
    width, height = args.size
    source = args.image or work / "contrôle.png"
    if args.image is None:
        control_image(source, width, height)
    options = {"prompt": args.prompt, "width": width, "height": height, "steps": args.steps, "seed": 42,
               "text_tokens": args.text_tokens, "guidance": 30.0, "canny_low": 50, "canny_high": 200}
    base_paths = dict(bin_dir=args.bin_dir, weights=args.weights, clip_tokenizer=args.clip_tokenizer,
                      t5_tokenizer=args.t5_tokenizer)
    expected = {"seed": options["seed"], "steps": args.steps, "decoder": "vae", "profile": "canny_dev",
                **{key: options[key] for key in ("guidance", "canny_low", "canny_high")}}
    summary = {}
    for mode in ("direct", "cli"):
        with scratch_directory() as case:
            env = local_environment(case)
            if mode == "direct":
                png = case / "flux.png"
                command = [sys.executable, "-B", HERE / "generate.py", "--output", png, "--workspace", case,
                           "--control-image", source, "--backend", "cuda", "--expected-device", args.expected_device]
                for key, value in {**base_paths, **options}.items():
                    command.extend(["--" + key.replace("_", "-"), str(value)])
            else:
                app = case / "ImageStudio/app"
                shutil.copytree(args.studio / "app", app, ignore=shutil.ignore_patterns("__pycache__", "packages"))
                config_path = case / "canny-runtime.json"
                config_paths = {"python": sys.executable, "launcher": HERE / "generate.py", **base_paths}
                config_path.write_text(json.dumps({**{key: str(path) for key, path in config_paths.items()},
                                                   "expected_cuda_device": args.expected_device}), encoding="utf-8")
                parameters = case / "request.json"
                parameters.write_text(json.dumps({**options, "image": str(source)}, ensure_ascii=False), encoding="utf-8")
                export = case / "export"
                env.update(IMAGESTUDIO_FLUX_CANNY_ZML_CONFIG=str(config_path),
                           CYTOOLS_SETTINGS_DIR=str(case / "cytools-settings"))
                env.pop("CYTOOLS_TOKEN", None)
                cli = [args.studio / "runtime/python/Scripts/python.exe", "-B", "-X", "utf8", app / "tool.py",
                       "--data-dir", case / "jobs"]
                client = subprocess.check_output([*cli, "create-client", "--name", "Canny validation",
                                                  "--permission", "filesystemExternal"], env=env, encoding="utf-8")
                env["CYTOOLS_TOKEN"] = json.loads(client)["token"]
                command = [*cli, "invoke", "generate_flux_canny_zml", "--backend", "zml-cuda",
                           "--params-file", parameters, "--export-dir", export, "--timeout", "1800"]
            print(f"Running Canny Dev {mode} at {width}x{height}, {args.steps} steps", flush=True)
            result, metrics = measure(command, env, case, args.gpu)
            if result.returncode:
                print(result.stdout, result.stderr, file=sys.stderr)
                result.check_returncode()
            report = json.loads(result.stdout)
            if mode == "cli":
                report = report["outputs"]
                png = Path(report["assets"][0]["file"])
                if not png.resolve(strict=True).is_relative_to(export):
                    raise AssertionError("CLI returned an image outside the test export")
                [log] = export.rglob("worker-stderr.log")
                diagnostics = log.read_text(encoding="utf-8")
            else:
                if Path(report["file"]) != png:
                    raise AssertionError("Direct launcher returned an unexpected output path")
                diagnostics = result.stderr
            placement = verify_placement(diagnostics, "cuda", args.steps + 5, args.expected_device)
            if placement != report["placement"] or any(report.get(key) != value for key, value in expected.items()):
                raise AssertionError(f"Unexpected {mode} metadata: {report}")
            summary[mode] = {"sha256": inspect_png(png, args.size), "placement": placement, "storage": report["storage"],
                             "verified_placements": args.steps + 5, **metrics}
            print(json.dumps({mode: summary[mode]}, allow_nan=False), flush=True)
    if summary["direct"]["sha256"] != summary["cli"]["sha256"]:
        raise AssertionError("CLI and direct PNG bytes differ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--weights", type=Path, help="Prepared Canny Dev bundle")
    sources.add_argument("--model-root", type=Path, help="ComfyUI models directory with the four standalone files")
    for name in ("bin-dir", "clip-tokenizer", "t5-tokenizer", "studio"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--size", type=image_size, default=(384, 384))
    parser.add_argument("--expected-device", default="RTX 3090")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--text-tokens", type=int, default=128)
    parser.add_argument("--prompt", default="A mountain in front of a golden sun, a lake and a white pier, illustration.")
    args = parser.parse_args()
    if not 1 <= args.steps <= 50 or not 2 <= args.text_tokens <= 512:
        parser.error("Use 1 to 50 steps and 2 to 512 T5 tokens")
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve(strict=True))
    with scratch_directory() as work:
        if args.model_root:
            args.weights = work / "bundle"
            sources = {
                "denoiser": "diffusion_models/flux1-canny-dev.safetensors",
                "clip": "text_encoders/clip_l.safetensors",
                "t5": "text_encoders/t5xxl_fp16.safetensors",
                "vae": "vae/ae.safetensors",
            }
            command = [sys.executable, "-B", HERE / "prepare.py", "--profile", "canny_dev",
                       "--bin-dir", args.bin_dir, "--output", args.weights]
            for key, file in sources.items():
                command.extend(["--" + key, args.model_root / file])
            subprocess.run(command, env=local_environment(work), check=True)
        validate(args, work)


if __name__ == "__main__":
    main()
