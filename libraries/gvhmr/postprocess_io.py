"""Feed decoded GVHMR motion and the local SMPL-X skeleton to the ZML corrections."""

import subprocess
import sys

import numpy as np
from safetensors.numpy import save_file
import torch

from probe_placement import verify_placement


PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)


def save_skeleton(endecoder, destination):
    body = endecoder.smplx_model
    rest = body.J_template[:22].detach().cpu().numpy()
    directions = body.J_shapedirs[:22].detach().cpu().numpy()
    if tuple(endecoder.parents) != PARENTS or rest.shape != (22, 3) or directions.shape != (22, 3, 10):
        raise ValueError("Invalid neutral SMPL-X skeleton for GVHMR")
    if any(value.dtype != np.float32 or not np.isfinite(value).all() for value in (rest, directions)):
        raise ValueError("Expected finite neutral SMPL-X skeleton tensors in FP32")
    save_file({"J_template": rest.reshape(66), "J_shapedirs": directions.reshape(66, 10).T.copy()}, destination,
              metadata={"architecture": "GVHMR neutral 22-joint SMPL-X skeleton", "storage": "F32"})


def pack(prediction):
    world = prediction["pred_smpl_params_global"]
    camera = prediction["pred_smpl_params_incam"]
    fields = (world["betas"], world["global_orient"], world["body_pose"], world["transl"],
              prediction["static_conf_logits"], camera["global_orient"], camera["transl"])
    frames = world["betas"].shape[1]
    for width, value in zip((10, 3, 63, 3, 6, 3, 3), fields):
        if value.shape != (1, frames, width) or value.dtype != torch.float32:
            raise ValueError("Expected a single decoded GVHMR sequence in FP32")
    values = torch.cat(fields, dim=-1)[0].detach().cpu().numpy().astype("<f4", copy=False)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite input to GVHMR corrections")
    return values


def run(probe, backend, camera, weights, prediction, work, *, repeats, expected_device):
    values = pack(prediction)
    source, target = work / "postprocess-in.f32", work / "postprocess-out.f32"
    values.tofile(source)
    result = subprocess.run([probe, backend, camera, weights, str(len(values)), source, target, str(repeats)],
                            stdout=sys.stderr, stderr=subprocess.PIPE, text=True)
    sys.stderr.write(result.stderr)
    result.check_returncode()
    verify_placement(result.stderr, backend, repeats, expected_device)
    output = np.fromfile(target, dtype="<f4").reshape(1, len(values), 66)
    if not np.isfinite(output).all():
        raise ValueError("Non-finite output from GVHMR corrections")
    return {"transl": torch.from_numpy(output[..., :3].copy()), "body_pose": torch.from_numpy(output[..., 3:].copy())}


def reference(prediction, endecoder, camera):
    from hmr4d.model.gvhmr.utils.postprocess import pp_static_joint, pp_static_joint_cam, process_ik

    staged = dict(prediction)
    staged["pred_smpl_params_global"] = dict(prediction["pred_smpl_params_global"])
    contact = pp_static_joint_cam if camera == "fixed" else pp_static_joint
    translation = contact(staged, endecoder)
    staged["pred_smpl_params_global"]["transl"] = translation
    return {"transl": translation, "body_pose": process_ik(staged, endecoder)}


def compare(actual, expected):
    result = {}
    for field, tolerance in (("transl", 3e-5), ("body_pose", 8e-5)):
        native = actual[field].detach().cpu().numpy()
        baseline = expected[field].detach().cpu().numpy()
        np.testing.assert_allclose(native, baseline, atol=tolerance, rtol=2e-5, equal_nan=False, err_msg=field)
        result[field] = float(np.abs(native - baseline).max())
    return result
