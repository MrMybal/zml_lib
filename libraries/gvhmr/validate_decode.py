"""Compare the ZML body decoder with the pinned GVHMR reference in FP32."""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

import torch

from video import use_local_runtime
from build import scratch_directory
from probe_placement import PLATFORMS
import decode_io


CASES = {"single": (1, False), "fixed137": (137, False), "moving137": (137, True)}


def example(endecoder, frames, moving):
    from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d

    def rotation6d(angles):
        return matrix_to_rotation_6d(axis_angle_to_matrix(angles))

    time = torch.linspace(0, 1, frames, dtype=torch.float32)
    zeros = torch.zeros_like(time)
    rng = torch.Generator().manual_seed(91)
    body = torch.randn(frames, 21, 3, generator=rng) * 0.28
    body[0, 0] = torch.tensor([0, 0, torch.pi - 1e-4])
    betas = torch.randn(frames, 10, generator=rng) * 0.16
    camera_root = torch.stack((0.11 + 0.14 * time, 0.32 * time.sin(), 0.17 * time), dim=-1)
    gravity_root = torch.stack((0.17 + 0.10 * time, 0.30 + 0.40 * time, 0.15 - 0.27 * time), dim=-1)
    velocity = torch.stack((0.017 * (1 + time), 0.003 * time.sin(), -0.025 + 0.007 * time), dim=-1)
    raw = torch.cat((rotation6d(body).reshape(frames, 126), betas, rotation6d(camera_root),
                     rotation6d(gravity_root), velocity), dim=-1)
    scale = torch.stack((1.2 + 0.11 * time, time.sin() * 0.18, time * -0.12), dim=-1)
    predictions = {"pred_x": ((raw - endecoder.mean) / endecoder.std)[None], "pred_cam": scale[None]}

    relative_yaw = (0.031 * (time * 12 + 0.7).sin()) if moving else zeros.clone()
    # The near-identity camera increments exercise GVHMR's 1e-5 rotation snap.
    relative_yaw[::9] = 2e-7
    relative = rotation6d(torch.stack((zeros, relative_yaw, zeros), dim=-1))
    intrinsics = torch.eye(3).repeat(frames, 1, 1)
    intrinsics[:, 0, 0] = 950 + 30 * time
    intrinsics[:, 1, 1] = intrinsics[:, 0, 0]
    intrinsics[:, 0, 2] = 540 + 15 * time
    intrinsics[:, 1, 2] = 330
    crops = torch.stack((520 + 55 * time, 315 + 24 * time, 420 + 50 * time), dim=-1)
    return predictions, {"bbx_xys": crops[None], "K_fullimg": intrinsics[None], "cam_angvel": relative[None]}


@torch.inference_mode()
def validate(args, work, endecoder):
    frames, moving = CASES[args.case]
    weights = work / "decode-stats.safetensors"
    decode_io.save_stats(endecoder, weights)
    model_output, batch = example(endecoder, frames, moving)
    output = decode_io.run(args.probe, args.backend, weights, model_output, batch, work,
                           repeats=2, expected_device=args.expected_device)
    expected = decode_io.reference(model_output, batch, endecoder)
    errors = decode_io.compare(output, expected)
    upstream = decode_io.reference(model_output, batch, endecoder, stable=False)
    torch.testing.assert_close(output["transl_world"][:, :1], torch.zeros(1, 1, 3), rtol=0, atol=0)
    if frames > 1 and output["transl_world"].abs().max().item() < 1:
        raise RuntimeError("The decoded sequence did not exercise integration of the local velocity")
    return {"case": args.case, "frames": frames, "backend": args.backend, "pjrt": PLATFORMS[args.backend],
            "inferences": 2, "max_abs_error": errors, "upstream_max_abs_error": decode_io.compare(output, upstream, verify=False)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("probe", "four-d-anyone", "upstream"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--case", choices=CASES, default="moving137")
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

            summary = validate(args, work, EnDecoder(stats_name="MM_V1_AMASS_LOCAL_BEDLAM_CAM").eval())
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
