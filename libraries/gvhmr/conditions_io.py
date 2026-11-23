"""Prepare the SIGA24 observations and camera conditions with the native graph."""

import subprocess
import sys

import numpy as np

from probe_placement import verify_placement


def run(probe, backend, keypoints, boxes, intrinsics, angular, work, *, stats=None, repeats=1, expected_device=None):
    frames = len(keypoints)
    fields = ((keypoints, (17, 3)), (boxes, (3,)), (intrinsics, (3, 3)), (angular, (6,)))
    if not 1 <= frames <= 4096 or any(value.shape != (frames, *shape) or value.dtype != np.float32
                                      for value, shape in fields):
        raise ValueError("Expected keypoints, boxes, intrinsics and orientations in FP32 for 1–4096 frames")
    if (boxes[:, 2] <= 0).any() or (intrinsics[:, 0, 0] <= 0).any():
        raise ValueError("Expected a positive crop side and focal length per frame")
    camera = intrinsics[:, [0, 0, 1], [0, 2, 2]]
    data = np.concatenate((keypoints.reshape(frames, 51), boxes, camera, angular), axis=1).astype("<f4", copy=False)
    if stats is None:
        calibration = np.array([0] * 6 + [1] * 6, dtype="<f4")
    else:
        mean, deviation = (np.asarray(value, dtype="<f4").reshape(6) for value in stats)
        calibration = np.concatenate((mean, deviation))
    if not np.isfinite(data).all() or not np.isfinite(calibration).all() or (calibration[6:] <= 0).any():
        raise ValueError("Expected finite FP32 inputs and positive camera standard deviations")

    source, stats_file, destination = (work / f"conditions-{name}.f32" for name in ("input", "stats", "output"))
    data.tofile(source)
    calibration.tofile(stats_file)
    command = [probe, backend, str(frames), source, stats_file, destination, str(repeats)]
    result = subprocess.run(command, stdout=sys.stderr, stderr=subprocess.PIPE, text=True)
    sys.stderr.write(result.stderr)
    result.check_returncode()
    verify_placement(result.stderr, backend, repeats, expected_device)
    output = np.fromfile(destination, dtype="<f4").reshape(frames, 60)
    if not np.isfinite(output).all():
        raise ValueError("Non-finite output from the GVHMR conditions graph")
    return {"obs": output[:, :51].reshape(frames, 17, 3), "f_cliffcam": output[:, 51:54],
            "f_cam_angvel": output[:, 54:]}


def compare(actual, expected):
    np.testing.assert_array_equal(actual["obs"][..., 2], expected["obs"][..., 2])
    errors = {}
    for field, value in actual.items():
        np.testing.assert_allclose(value, expected[field], rtol=2e-6, atol=2e-6, equal_nan=False, err_msg=field)
        errors[field] = float(np.max(np.abs(value - expected[field])))
    return errors
