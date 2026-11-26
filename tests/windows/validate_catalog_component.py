"""Prepare one local ComfyUI component and compare ZML CUDA with PyTorch."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.build import run_environment, scratch_directory
from scripts.model_catalog import COMPONENTS, identify, prepare, require_probe, tensor_header
from scripts.probe_placement import verify_placement


def clip_case(checkpoint, weights, work):
    import torch
    from safetensors.torch import load_file
    from transformers import CLIPTextModel

    from libraries.stable_diffusion.prepare import clip_config

    ids = [49406, 320, 1000, 49407]
    model = CLIPTextModel(clip_config()).eval()
    state = load_file(checkpoint)
    state.pop("text_model.embeddings.position_ids", None)
    model.load_state_dict(state)
    expected = model(input_ids=torch.tensor([ids])).last_hidden_state.numpy()
    output = work / "clip.f32"
    return ["cuda", weights, output, *map(str, ids)], output, expected, 1, (2e-4, 1e-3)


def t5_case(checkpoint, weights, probe, work):
    sys.path.insert(0, str(ROOT / "libraries/t5"))
    from validate import cases_from_transformers, schema_for

    length = 9
    schema = schema_for(probe, "t5_xxl")
    [(inputs, expected)] = cases_from_transformers("t5_xxl", checkpoint, schema, (length,), None)
    input_path = work / "tokens.f32"
    output = work / "t5.f32"
    inputs.tofile(input_path)
    command = ["t5_xxl", "--storage=f16", "cuda", weights, str(length), input_path, output]
    return command, output, expected, 2, (4e-4, 3e-4)


def vae_case(checkpoint, weights, work):
    import numpy as np
    import torch
    from diffusers import AutoencoderKL
    from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
    from safetensors.torch import load_file

    with torch.device("meta"):
        vae = AutoencoderKL(
            block_out_channels=(128, 256, 512, 512), latent_channels=16, layers_per_block=2,
            down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4,
            scaling_factor=0.3611, shift_factor=0.1159, use_quant_conv=False, use_post_quant_conv=False,
        ).eval()
    state = convert_ldm_vae_checkpoint(load_file(checkpoint), vae.config)
    state = {name: value.float() for name, value in state.items()}
    vae.load_state_dict(state, strict=True, assign=True)
    latent = np.random.default_rng(37).standard_normal((1, 16, 4, 6), dtype=np.float32)
    expected = vae.decode(torch.from_numpy(latent) / vae.config.scaling_factor + vae.config.shift_factor).sample.numpy()
    input_path = work / "latent.f32"
    output = work / "rgb.f32"
    latent.tofile(input_path)
    return ["vae", "cuda", weights, "4", "6", input_path, output], output, expected, 1, (6e-4, 6e-4)


def validate(checkpoint, bin_dir, expected_device, work):
    import numpy as np
    import torch

    torch.set_num_threads(8)
    match = identify(checkpoint)
    if match.profile not in COMPONENTS:
        raise ValueError(f"Expected a standalone model component, got {match.profile}")
    library, name, _ = COMPONENTS[match.profile]
    probe = require_probe(bin_dir, library, name)
    print(f"Preparing {match.profile}: {checkpoint.name}", flush=True)
    report = prepare(checkpoint, work / "prepared", bin_dir)
    weights = Path(report["path"])
    assert identify(weights).ready, "Prepared weights were not recognized"
    if not match.ready:
        assert report["status"] == "prepared" and weights.is_relative_to(work), "Expected a fresh cache entry"
        cached = prepare(checkpoint, work / "prepared", bin_dir)
        assert cached["status"] == "cached" and cached["path"] == report["path"], "Cache was not reused"
    print(f"Comparing {match.profile} with PyTorch", flush=True)
    with torch.inference_mode():
        if match.profile == "clip_l":
            case = clip_case(checkpoint, weights, work)
        elif match.profile == "t5_xxl":
            case = t5_case(checkpoint, weights, probe, work)
        else:
            case = vae_case(checkpoint, weights, work)
    args, output, expected, repeats, (atol, rtol) = case
    result = subprocess.run([probe, *args], capture_output=True, text=True, errors="replace")
    print(result.stderr, end="", file=sys.stderr)
    result.check_returncode()
    placement = verify_placement(result.stderr, "cuda", repeats, expected_device)
    observed = np.fromfile(output, dtype="<f4").reshape(expected.shape)
    delta = np.abs(observed - expected)
    print(json.dumps({"profile": match.profile, "output_shape": list(expected.shape), "placement": placement,
                      "precision": match.precision, "tensors": len(tensor_header(weights)),
                      "max_abs": float(delta.max()), "rmse": float(np.sqrt(np.mean(delta ** 2)))},
                     allow_nan=False), flush=True)
    np.testing.assert_allclose(observed, expected, atol=atol, rtol=rtol, equal_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--expected-device", default="RTX 3090")
    args = parser.parse_args()
    bin_dir = args.bin_dir.resolve(strict=True)
    tools = ROOT.parent / "zml_win" / "bazel-win-tools"
    for checkpoint in args.checkpoints:
        with scratch_directory() as work:
            os.environ.update(run_environment(work, tools))
            os.environ.update(TORCHINDUCTOR_CACHE_DIR=str(work / "inductor"), TRITON_CACHE_DIR=str(work / "triton"),
                              CUDA_CACHE_DISABLE="1", HF_HUB_OFFLINE="1")
            validate(checkpoint.resolve(strict=True), bin_dir, args.expected_device, work)


if __name__ == "__main__":
    main()
