"""Pass SIGA24 predictions and camera geometry to the native body decoder."""

import subprocess
import sys

import numpy as np
from safetensors.numpy import save_file
import torch

from probe_placement import verify_placement


MOTION_FIELDS = ("body_pose", "betas", "global_orient", "global_orient_gv", "local_transl_vel")
OUTPUT_FIELDS = tuple(zip(MOTION_FIELDS, (63, 10, 3, 3, 3))) + (
    ("transl_incam", 3), ("global_orient_world", 3), ("transl_world", 3),
)


def save_stats(endecoder, destination):
    weights = {key: getattr(endecoder, key).detach().cpu().numpy() for key in ("mean", "std")}
    for value in weights.values():
        if value.shape != (151,) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("Expected the 151-channel GVHMR SIGA24 FP32 decode statistics")
    if np.any(weights["std"] <= 0):
        raise ValueError("Invalid GVHMR motion standard deviation")
    save_file(weights, destination, metadata={"architecture": "GVHMR SIGA24 body decode", "storage": "F32"})


def temporal_conditions(pipeline, batch):
    from hmr4d.utils.geo.hmr_cam import compute_bbox_info_bedlam

    angular = batch["cam_angvel"]
    if pipeline.args.normalize_cam_angvel:
        angular = (angular - pipeline.cam_angvel_mean) / pipeline.cam_angvel_std
    return {
        "obs": batch["obs"], "f_imgseq": batch["f_imgseq"], "f_cam_angvel": angular,
        "f_cliffcam": compute_bbox_info_bedlam(batch["bbx_xys"], batch["K_fullimg"]),
    }


def pack(model_output, batch):
    x = model_output["pred_x"]
    intrinsics = batch["K_fullimg"]
    frames = x.shape[1]
    if not 1 <= frames <= 4096 or intrinsics.shape != (1, frames, 3, 3):
        raise ValueError("Expected camera intrinsics for one GVHMR sequence")
    camera = torch.stack((intrinsics[..., 0, 0], intrinsics[..., 0, 2], intrinsics[..., 1, 2]), dim=-1)
    fields = (x, model_output["pred_cam"], batch["bbx_xys"], camera, batch["cam_angvel"])
    for width, value in zip((151, 3, 3, 3, 6), fields):
        if value.shape != (1, frames, width) or value.dtype != torch.float32:
            raise ValueError("Expected FP32 GVHMR predictions, crops, intrinsics and camera rotations")
    values = torch.cat(fields, dim=-1)[0].detach().cpu().numpy().astype("<f4", copy=False)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite input to the GVHMR body decoder")
    return values


def run(probe, backend, weights, model_output, batch, work, *, repeats, expected_device):
    values = pack(model_output, batch)
    source, target = work / "decode-in.f32", work / "decode-out.f32"
    values.tofile(source)
    result = subprocess.run([probe, backend, weights, str(len(values)), source, target, str(repeats)],
                            stdout=sys.stderr, stderr=subprocess.PIPE, text=True)
    sys.stderr.write(result.stderr)
    result.check_returncode()
    verify_placement(result.stderr, backend, repeats, expected_device)
    output = np.fromfile(target, dtype="<f4").reshape(1, len(values), sum(n for _, n in OUTPUT_FIELDS))
    if not np.isfinite(output).all():
        raise ValueError("Non-finite output from the GVHMR body decoder")
    decoded = {}
    offset = 0
    for key, width in OUTPUT_FIELDS:
        decoded[key] = torch.from_numpy(output[..., offset:offset + width].copy())
        offset += width
    return decoded


def assemble(model_output, decoded):
    common = {key: decoded[key] for key in ("body_pose", "betas")}
    return {
        "model_output": model_output,
        "decode_dict": {key: decoded[key] for key in MOTION_FIELDS},
        "static_conf_logits": model_output["static_conf_logits"],
        "pred_smpl_params_incam": {**common, "global_orient": decoded["global_orient"], "transl": decoded["transl_incam"]},
        "pred_smpl_params_global": {**common, "global_orient": decoded["global_orient_world"], "transl": decoded["transl_world"]},
    }


def stable_world_reference(decoded, angular):
    from hmr4d.utils.geo.hmr_global import get_tgtcoord_rootparam, rollout_local_transl_vel
    from hmr4d.utils.net_utils import gaussian_smooth
    from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, rotation_6d_to_matrix
    from torch.nn.functional import normalize

    relative = rotation_6d_to_matrix(angular)
    identity = torch.eye(3, dtype=relative.dtype, device=relative.device)
    small = matrix_to_axis_angle(relative).norm(dim=-1) < 1e-5
    relative = torch.where(small[..., None, None], identity, relative)
    gravity = axis_angle_to_matrix(decoded["global_orient_gv"])
    camera = axis_angle_to_matrix(decoded["global_orient"])
    view = gravity @ camera.mT
    horizontal = normalize(view[..., (0, 2), 2], dim=-1)
    next_horizontal = normalize((view @ relative.mT)[..., (0, 2), 2], dim=-1)
    cross = next_horizontal[..., 1] * horizontal[..., 0] - next_horizontal[..., 0] * horizontal[..., 1]
    dot = (horizontal * next_horizontal).sum(dim=-1).clamp(-1, 1)
    yaw = torch.atan2(cross.abs(), dot) * (cross / cross.abs().clamp_min(1e-12))
    yaw = torch.where(small, 0, yaw)
    increments = axis_angle_to_matrix(gaussian_smooth(yaw, dim=-1)[..., None] * angular.new_tensor((0, 1, 0))).mT
    # Sequential composition keeps this reference independent of the ZML parallel scan.
    history = [identity.expand(angular.shape[0], -1, -1)]
    for frame in range(1, angular.shape[1]):
        history.append(history[-1] @ increments[:, frame])
    history = torch.stack(history, dim=1)
    small = matrix_to_axis_angle(history).norm(dim=-1) < 1e-5
    history = torch.where(small[..., None, None], identity, history)
    root = matrix_to_axis_angle(history @ gravity)
    translation = rollout_local_transl_vel(decoded["local_transl_vel"], root)
    root, translation, _ = get_tgtcoord_rootparam(root, translation, tsf="any->ay")
    return {"global_orient": root, "transl": translation}


def reference(model_output, batch, endecoder, *, stable=True):
    from hmr4d.model.gvhmr.pipeline.gvhmr_pipeline import get_smpl_params_w_Rt_v2
    from hmr4d.utils.geo.hmr_cam import compute_transl_full_cam

    decoded = endecoder.decode(model_output["pred_x"])
    if stable:
        world = stable_world_reference(decoded, batch["cam_angvel"])
    else:
        world = get_smpl_params_w_Rt_v2(decoded["global_orient_gv"], decoded["local_transl_vel"],
                                        decoded["global_orient"], batch["cam_angvel"])
    return {
        **decoded,
        "transl_incam": compute_transl_full_cam(model_output["pred_cam"], batch["bbx_xys"], batch["K_fullimg"]),
        "global_orient_world": world["global_orient"], "transl_world": world["transl"],
    }


def compare(actual, expected, *, verify=True):
    from pytorch3d.transforms import axis_angle_to_matrix

    errors = {}
    rotations = ("body_pose", "global_orient", "global_orient_gv", "global_orient_world")
    for key, _ in OUTPUT_FIELDS:
        a, b = actual[key], expected[key]
        tolerance = 2e-4 if key in ("global_orient_world", "transl_world") else 2e-5
        label = key
        if key in rotations:
            a = axis_angle_to_matrix(a.reshape(-1, 3))
            b = axis_angle_to_matrix(b.reshape(-1, 3))
            label += "_matrix"
        a, b = a.detach().cpu().numpy(), b.detach().cpu().numpy()
        if verify:
            np.testing.assert_allclose(a, b, atol=tolerance, rtol=2e-5, equal_nan=False, err_msg=label)
        errors[label] = float(np.abs(a - b).max())
    return errors
