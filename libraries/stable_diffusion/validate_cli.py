"""Compare direct SD 1.5 generation or inpainting with the ImageStudio CLI on CUDA."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

from generate import scratch_directory
from inpaint_fixture import write_fixture
from prepare import prepare
from validate import set_validation_caches


def validate(args, work):
    launcher = Path(__file__).with_name("inpaint.py" if args.inpaint else "generate.py").resolve()
    profile = "sd15-inpaint" if args.inpaint else "sd15"
    weights = args.weights
    if weights is None:
        weights = work / "weights"
        prepare(args.checkpoint, args.bin_dir, weights, profile)
    config = work / "sd15-runtime.json"
    config.write_text(json.dumps({
        "python": sys.executable, "launcher": str(launcher), "bin_dir": str(args.bin_dir),
        "weights": str(weights), "clip_tokenizer": str(args.clip_tokenizer),
        "expected_cuda_device": args.expected_device,
    }), encoding="utf-8")
    parameters = {"prompt": "A red lighthouse by the sea at dawn", "negative": "blur", "seed": 42,
                  "steps": 4, "guidance": 7.5}
    if args.inpaint:
        image_path, mask_path = write_fixture(work, args.width, args.height)
        parameters.update(image=str(image_path), mask=str(mask_path), strength=args.strength)
    else:
        parameters.update(width=args.width, height=args.height)
    params = work / "parameters.json"
    params.write_text(json.dumps(parameters), encoding="utf-8")
    env = os.environ.copy()
    for key in ("CYMODELHUB_SERVICE", "CYTOOLS_TOKEN"):
        env.pop(key, None)
    for key in ("TEMP", "TMP", "TMPDIR", "HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME",
                "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        env[key] = str(work)
    env.update(CYMODELHUB_DATA=str(work / "model-hub"), CYTOOLS_SETTINGS_DIR=str(work / "settings"),
               PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1")
    env["IMAGESTUDIO_SD15_INPAINT_ZML_CONFIG" if args.inpaint else "IMAGESTUDIO_SD15_ZML_CONFIG"] = str(config)
    # ImageStudio also resolves its task input storage relative to app/tool.py.
    staged_app = work / "app"
    shutil.copytree(args.studio_app, staged_app, ignore=shutil.ignore_patterns("__pycache__", ".git", ".venv"))
    cli_base = [args.studio_python, "-B", staged_app / "tool.py", "--data-dir", work / "jobs"]
    cli_env = env.copy()
    if args.inpaint:
        client = subprocess.check_output(
            [*cli_base, "create-client", "--name", "SD 1.5 Inpainting validation", "--permission", "filesystemExternal"],
            cwd=work, env=env, encoding="utf-8",
        )
        cli_env["CYTOOLS_TOKEN"] = json.loads(client)["token"]

    direct = work / "direct.png"
    command = [sys.executable, "-B", launcher, "--weights", weights, "--bin-dir", args.bin_dir,
               "--clip-tokenizer", args.clip_tokenizer, "--backend", "cuda", "--expected-device", args.expected_device,
               "--workspace", work, "--output", direct, "--negative-prompt", parameters["negative"]]
    for key, value in parameters.items():
        if key != "negative":
            command.extend(("--" + key, str(value)))
    direct_result = json.loads(subprocess.check_output(command, env=env, encoding="utf-8"))
    export = work / "export"
    operation = "inpaint_sd15_zml" if args.inpaint else "generate_sd15_zml"
    cli = subprocess.run([*cli_base, "invoke", operation, "--params-file", params, "--export-dir", export,
                          "--timeout", "900", "--backend", "zml-cuda"], cwd=work, env=cli_env, encoding="utf-8",
                         stdout=subprocess.PIPE)
    for log in export.rglob("worker-stderr.log"):
        print(log.read_text(encoding="utf-8", errors="replace"), end="", file=sys.stderr)
    cli.check_returncode()
    report = json.loads(cli.stdout)
    output = Path(report["directory"]) / "outputs" / (profile + ".png")
    for file in (direct, output):
        with Image.open(file) as image:
            if (image.format, image.mode, image.size) != ("PNG", "RGB", (args.width, args.height)):
                raise ValueError(f"Unexpected output image: {file}")
            image.verify()
    if report["outputs"]["placement"] != direct_result["placement"]:
        raise ValueError("The launcher and ImageStudio reported different device placements.")
    if output.read_bytes() != direct.read_bytes():
        raise ValueError("The launcher and ImageStudio PNGs differ.")
    result = {"sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "width": args.width, "height": args.height, "placement": direct_result["placement"]}
    if args.inpaint:
        with Image.open(image_path) as source, Image.open(mask_path) as mask, Image.open(output) as decoded:
            region = np.asarray(mask) >= 128
            edited = np.any(np.asarray(source) != np.asarray(decoded), axis=2)
        if np.any(edited & ~region) or not np.any(edited & region):
            raise ValueError("The PNG must edit the mask and preserve all pixels outside it.")
        for key in ("strength", "effective_steps"):
            if report["outputs"][key] != direct_result[key]:
                raise ValueError(f"ImageStudio reported a different {key}.")
        result.update(changed_inside=int(edited.sum()), changed_outside=0, strength=direct_result["strength"],
                      effective_steps=direct_result["effective_steps"])
    return result


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
    parser.add_argument("--inpaint", action="store_true")
    parser.add_argument("--strength", type=float, default=0.75)
    args = parser.parse_args()
    for key in ("checkpoint", "weights", "bin_dir", "clip_tokenizer", "studio_app", "studio_python"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.resolve(strict=True))
    with scratch_directory() as work:
        set_validation_caches(work)
        result = validate(args, work)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
