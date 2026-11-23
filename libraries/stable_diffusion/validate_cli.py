"""Compare direct SD 1.5 generation with the CyTools ImageStudio CLI on CUDA."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image

from generate import scratch_directory
from prepare import prepare


def validate(args, work):
    launcher = Path(__file__).with_name("generate.py").resolve()
    weights = args.weights
    if weights is None:
        weights = work / "weights"
        prepare(args.checkpoint, args.bin_dir, weights)
    config = work / "sd15-runtime.json"
    config.write_text(json.dumps({
        "python": sys.executable, "launcher": str(launcher), "bin_dir": str(args.bin_dir),
        "weights": str(weights), "clip_tokenizer": str(args.clip_tokenizer),
        "expected_cuda_device": args.expected_device,
    }), encoding="utf-8")
    parameters = {"prompt": "A red lighthouse by the sea at dawn", "negative": "blur", "seed": 42,
                  "width": args.width, "height": args.height, "steps": 4, "guidance": 7.5}
    params = work / "parameters.json"
    params.write_text(json.dumps(parameters), encoding="utf-8")
    env = os.environ.copy()
    for key in ("CYMODELHUB_SERVICE", "CYTOOLS_TOKEN"):
        env.pop(key, None)
    for key in ("TEMP", "TMP", "TMPDIR", "HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME",
                "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        env[key] = str(work)
    env.update(CYMODELHUB_DATA=str(work / "model-hub"), CYTOOLS_SETTINGS_DIR=str(work / "settings"),
               IMAGESTUDIO_SD15_ZML_CONFIG=str(config), PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1")

    direct = work / "direct.png"
    command = [sys.executable, "-B", launcher, "--weights", weights, "--bin-dir", args.bin_dir,
               "--clip-tokenizer", args.clip_tokenizer, "--backend", "cuda", "--expected-device", args.expected_device,
               "--workspace", work, "--output", direct, "--negative-prompt", parameters["negative"]]
    for key in ("prompt", "width", "height", "steps", "guidance", "seed"):
        command.extend(("--" + key, str(parameters[key])))
    direct_result = json.loads(subprocess.check_output(command, env=env, encoding="utf-8"))
    export = work / "export"
    cli = subprocess.run([args.studio_python, "-B", args.studio_app / "tool.py", "--data-dir", work / "jobs",
                          "invoke", "generate_sd15_zml", "--params-file", params, "--export-dir", export,
                          "--timeout", "900", "--backend", "zml-cuda"], cwd=work, env=env, encoding="utf-8",
                         stdout=subprocess.PIPE)
    for log in export.rglob("worker-stderr.log"):
        print(log.read_text(encoding="utf-8", errors="replace"), end="", file=sys.stderr)
    cli.check_returncode()
    report = json.loads(cli.stdout)
    output = Path(report["directory"]) / "outputs/sd15.png"
    for file in (direct, output):
        with Image.open(file) as image:
            if (image.format, image.mode, image.size) != ("PNG", "RGB", (args.width, args.height)):
                raise ValueError(f"Unexpected output image: {file}")
            image.verify()
    if report["outputs"]["placement"] != direct_result["placement"]:
        raise ValueError("The launcher and ImageStudio reported different device placements.")
    if output.read_bytes() != direct.read_bytes():
        raise ValueError("The launcher and ImageStudio PNGs differ.")
    return {"sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "width": args.width, "height": args.height, "placement": direct_result["placement"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--weights", type=Path)
    for name in ("bin-dir", "clip-tokenizer", "studio-app", "studio-python"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--expected-device", default="RTX 3090")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    args = parser.parse_args()
    for key in ("checkpoint", "weights", "bin_dir", "clip_tokenizer", "studio_app", "studio_python"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.resolve(strict=True))
    with scratch_directory() as work:
        result = validate(args, work)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
