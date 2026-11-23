"""Compare the standalone neutral SMPL-X ZML mesh and joints with PyTorch FP32."""

import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from smplx_io import BATCH_SIZE, FIELDS, JOINTS, convert, pack, run
from build import ROOT, run_environment, scratch_directory
from probe_placement import PLATFORMS


@torch.inference_mode()
def validate(args, work):
    from smplx import SMPLX

    # One partial batch tests that padding is discarded and all passes keep their order.
    frames = BATCH_SIZE + 1
    rng = np.random.default_rng(117)
    params = {key: rng.normal(0, 0.2, size=(frames, width)).astype(np.float32) for key, width in FIELDS.items()}
    for array in params.values():
        array[:2] = 0
    params["transl"][1] = [1, -2, 0.75]
    params["global_orient"][2] = [1e-7, -2e-7, 1e-7]
    params["global_orient"][-1] = [0, np.pi - 1e-6, 0]

    target = work / "smplx.safetensors"
    faces = convert(args.checkpoint, target)
    reference = SMPLX(str(args.checkpoint), num_betas=10, num_expression_coeffs=10,
                      num_pca_comps=12, flat_hand_mean=False).eval()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    expected = reference(**{key: torch.from_numpy(value) for key, value in params.items()})
    np.testing.assert_array_equal(faces, reference.faces)

    verts, joints, dispatches = run(args.probe, args.backend, target, pack(params), work,
                                    repeats=2, expected_device=args.expected_device)
    errors = {}
    for key, actual, expected_tensor in (("vertices", verts, expected.vertices),
                                          ("kinematic_joints", joints, expected.joints[:, :JOINTS])):
        expected_values = expected_tensor.numpy()
        np.testing.assert_allclose(actual, expected_values, atol=3e-5, rtol=2e-5, equal_nan=False)
        np.testing.assert_allclose(actual[1] - actual[0], np.broadcast_to(params["transl"][1], actual.shape[1:]),
                                   atol=2e-6, rtol=0)
        errors[key] = float(np.abs(actual - expected_values).max())
    return {"frames": frames, "triangles": len(faces), "backend": args.backend,
            "pjrt": PLATFORMS[args.backend], "dispatches": dispatches, "max_abs_error": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backend", choices=PLATFORMS, default="cpu")
    parser.add_argument("--expected-device")
    args = parser.parse_args()
    args.probe = args.probe.resolve(strict=True)
    args.checkpoint = args.checkpoint.resolve(strict=True)
    with redirect_stdout(sys.stderr), scratch_directory() as work:
        os.environ.update(run_environment(work, ROOT.parent / "zml_win" / "bazel-win-tools"))
        summary = validate(args, work)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
