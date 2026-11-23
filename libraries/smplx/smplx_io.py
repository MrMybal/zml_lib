"""Convert a local neutral SMPL-X model and run its ZML mesh evaluator."""

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
from safetensors.numpy import save

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from probe_placement import verify_placement


VERTICES, JOINTS, TRIANGLES, BATCH_SIZE = 10475, 55, 20908, 16
ASSET = Path("inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz")
PARENTS = (
    0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
    15, 15, 15, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38,
    21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53,
)
FIELDS = {
    "betas": 10,
    "expression": 10,
    "global_orient": 3,
    "body_pose": 63,
    "jaw_pose": 3,
    "leye_pose": 3,
    "reye_pose": 3,
    "left_hand_pose": 12,
    "right_hand_pose": 12,
    "transl": 3,
}
INPUT_WIDTH = sum(FIELDS.values())


def as_f32(value, shape, name):
    if value.shape != shape or value.dtype.kind != "f":
        raise ValueError(f"Expected floating-point {name} with shape {shape}")
    array = np.ascontiguousarray(value, dtype="<f4")
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite {name}")
    return array


def read_model(source):
    with np.load(source, allow_pickle=False) as data:
        tree = data["kintree_table"]
        if (tree.shape != (2, JOINTS) or tree.dtype.kind not in "iu" or tree[0, 0] not in (-1, 4294967295)
                or not np.array_equal(tree[0, 1:], PARENTS) or not np.array_equal(tree[1], np.arange(JOINTS))):
            raise ValueError("Expected the SMPL-X kinematic tree for 55 joints")
        faces = data["f"]
        if (faces.shape != (TRIANGLES, 3) or faces.dtype.kind not in "iu"
                or faces.min() < 0 or faces.max() >= VERTICES):
            raise ValueError("Invalid SMPL-X face indices")
        shape = data["shapedirs"]
        if shape.shape != (VERTICES, 3, 400):
            raise ValueError("Expected 300 SMPL-X shape and 100 expression directions")
        shape = np.concatenate((shape[:, :, :10], shape[:, :, 300:310]), axis=2)
        pose = as_f32(data["posedirs"], (VERTICES, 3, 486), "posedirs")
        left_mean = as_f32(data["hands_meanl"], (45,), "hands_meanl")
        right_mean = as_f32(data["hands_meanr"], (45,), "hands_meanr")
        weights = {
            "v_template": as_f32(data["v_template"], (VERTICES, 3), "v_template").reshape(-1),
            "shapedirs": as_f32(shape, (VERTICES, 3, 20), "shapedirs").reshape(-1, 20).T.copy(),
            "posedirs": pose.reshape(-1, 486).T.copy(),
            "J_regressor": as_f32(data["J_regressor"], (JOINTS, VERTICES), "J_regressor"),
            "lbs_weights": as_f32(data["weights"], (VERTICES, JOINTS), "weights"),
            "left_hand_components": as_f32(data["hands_componentsl"], (45, 45), "hands_componentsl")[:12],
            "right_hand_components": as_f32(data["hands_componentsr"], (45, 45), "hands_componentsr")[:12],
            "pose_mean": np.concatenate((np.zeros(75, dtype="<f4"), left_mean, right_mean)),
        }
    return weights, faces.astype(np.int32)


def convert(source, destination):
    weights, faces = read_model(source)
    data = save(weights, metadata={"architecture": "SMPL-X neutral, GVHMR/SuperMotion", "storage": "F32"})
    with destination.open("xb") as stream:
        stream.write(data)
    return faces


def pack(params):
    if "betas" not in params or params.keys() - FIELDS.keys():
        raise ValueError("Expected SMPL-X betas and optional pose, expression and translation")
    frames = len(params["betas"])
    if not 1 <= frames <= 8192:
        raise ValueError("Expected 1–8192 SMPL-X parameter rows")
    return np.concatenate([
        as_f32(params[name], (frames, width), name) if name in params else np.zeros((frames, width), dtype="<f4")
        for name, width in FIELDS.items()
    ], axis=1)


def run(probe, backend, weights, packed, work, *, repeats=1, expected_device=None):
    if packed.ndim != 2 or packed.shape[1] != INPUT_WIDTH or not 1 <= len(packed) <= 8192:
        raise ValueError("Invalid packed SMPL-X input")
    source, target = work / "smplx-input.f32", work / "smplx-output.f32"
    packed.tofile(source)
    result = subprocess.run([probe, backend, weights, str(len(packed)), source, target, str(repeats)],
                            stdout=sys.stderr, stderr=subprocess.PIPE, text=True)
    sys.stderr.write(result.stderr)
    result.check_returncode()
    dispatches = ((len(packed) + BATCH_SIZE - 1) // BATCH_SIZE) * repeats
    verify_placement(result.stderr, backend, dispatches, expected_device)
    points = np.fromfile(target, dtype="<f4").reshape(len(packed), VERTICES + JOINTS, 3)
    if not np.isfinite(points).all():
        raise ValueError("Non-finite SMPL-X coordinates")
    return points[:, :VERTICES], points[:, VERTICES:], dispatches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("weights", type=Path)
    args = parser.parse_args()
    convert(args.checkpoint, args.weights)
