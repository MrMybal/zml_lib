"""Compare ZML contact and inverse kinematics against the pinned GVHMR implementation."""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

import torch

from video import use_local_runtime
from build import scratch_directory
from probe_placement import PLATFORMS
import postprocess_io


CASES = {"moving1": ("moving", 1), "moving27": ("moving", 27), "fixed27": ("fixed", 27)}


def example(frames):
    t = torch.linspace(0, 1, frames, dtype=torch.float32)
    rng = torch.Generator().manual_seed(723)
    pose = torch.randn(1, frames, 63, generator=rng) * 0.13
    betas = torch.randn(1, frames, 10, generator=rng) * 0.25
    world = torch.stack((1.4 * t, 0.92 + 0.2 * t.sin(), 0.3 * t), dim=1)[None]
    camera = torch.stack((0.2 * t, 1 + 0.12 * t, 3 + 0.2 * t), dim=1)[None]
    orient = torch.stack((0.08 * t, 1.3 * t, 0.13 * t), dim=1)[None]
    logits = torch.randn(1, frames, 6, generator=rng) * 2
    logits[:, ::5] = -2
    logits[:, 1::5, :4] = 3
    return {
        "pred_smpl_params_global": dict(body_pose=pose, betas=betas, global_orient=orient, transl=world),
        "pred_smpl_params_incam": dict(body_pose=pose, betas=betas, global_orient=orient * 0.5, transl=camera),
        "static_conf_logits": logits,
    }


@torch.inference_mode()
def validate(args, work, endecoder):
    camera, frames = CASES[args.case]
    weights = work / "skeleton.safetensors"
    postprocess_io.save_skeleton(endecoder, weights)
    prediction = example(frames)
    actual = postprocess_io.run(args.probe, args.backend, camera, weights, prediction, work,
                                repeats=2, expected_device=args.expected_device)
    expected = postprocess_io.reference(prediction, endecoder, camera)
    errors = postprocess_io.compare(actual, expected)
    original = prediction["pred_smpl_params_global"]
    update = {field: float((value - original[field]).abs().max()) for field, value in actual.items()}
    if update["transl"] <= 0.01 or (frames > 1 and update["body_pose"] <= 0.01):
        raise RuntimeError("The structured input did not exercise the contact/IK corrections")
    return {"case": args.case, "camera": camera, "frames": frames, "backend": args.backend,
            "pjrt": PLATFORMS[args.backend], "inferences": 2, "max_abs_error": errors, "max_abs_update": update}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("probe", "four-d-anyone", "upstream"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--case", choices=CASES, default="moving27")
    parser.add_argument("--backend", choices=PLATFORMS, default="cpu")
    parser.add_argument("--expected-device")
    args = parser.parse_args()
    args.probe = args.probe.resolve(strict=True)
    sys.path.insert(0, str(args.four_d_anyone.resolve(strict=True)))
    with redirect_stdout(sys.stderr), scratch_directory() as work:
        use_local_runtime(work)
        from fdanyone.motion import gvhmr as bridge

        upstream, _ = bridge.validate_gvhmr(args.upstream)
        with bridge.gvhmr_imports(upstream):
            bridge.install_pytorch3d_compat()
            bridge._register_inference_store()
            from hmr4d.model.gvhmr.utils.endecoder import EnDecoder

            summary = validate(args, work, EnDecoder().eval())
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
