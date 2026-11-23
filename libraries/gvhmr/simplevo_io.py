"""Feed sampled SIFT correspondences to the ZML two-view and trajectory graphs."""

from copy import deepcopy
import subprocess
import sys

import numpy as np

from probe_placement import verify_placement


HOMOGRAPHY_TRIALS, ESSENTIAL_TRIALS = 128, 96
SAMPLE_SIZE, PAIR_WIDTH, TRAJECTORY_WIDTH = 8, 15, 18


def calibration_for(height, width):
    # GVHMR uses a 24 mm focal length and the diagonal of a 36 × 24 mm sensor.
    image_diagonal = (width ** 2 + height ** 2) ** 0.5
    full_frame_diagonal = (36 ** 2 + 24 ** 2) ** 0.5
    return width, height, image_diagonal / full_frame_diagonal * 24


def keyframe_indices(frames):
    if frames < 2:
        raise ValueError("SimpleVO requires at least two frames")
    keys = np.arange(0, frames, 8, dtype=np.int32)
    if keys[-1] != frames - 1:
        keys = np.append(keys, np.int32(frames - 1))
    return keys


def keyframes(video, frames):
    from hmr4d.utils.video_io_utils import read_video_np

    keys = keyframe_indices(frames)
    images = read_video_np(str(video), scale=0.5)
    if len(images) != frames:
        raise ValueError("SimpleVO requires the canonical clip")
    return keys, images[keys], calibration_for(*images.shape[1:3])


def opencv_pairs(images, *, seed=None):
    from hmr4d.utils.preproc.relpose.matcher_wrapper import Matcher

    if seed is not None:
        import cv2

        cv2.setRNGSeed(seed)
    matcher = Matcher("sift")
    return [matcher.match_np(a, b) for a, b in zip(images[:-1], images[1:])]


def collect(video, frames):
    keys, images, calibration = keyframes(video, frames)
    return keys, opencv_pairs(images), calibration


def plan(keys):
    keys = np.asarray(keys, dtype=np.int32)
    if keys.ndim != 1 or len(keys) < 2 or keys[0] != 0 or keys[-1] >= 4096 or np.any(np.diff(keys) <= 0):
        raise ValueError("Invalid SimpleVO keyframe sequence")
    time = np.arange(keys[-1] + 1)
    right = np.searchsorted(keys, time)
    left = np.where(keys[right] == time, right, right - 1)
    fraction = (time - keys[left]) / np.maximum(1, keys[right] - keys[left])
    return np.stack((left, right, fraction), axis=1).astype("<f4")


def pack(pairs, calibration, tolerance):
    width, height, focal = calibration
    center = np.array([width / 2, height / 2])
    limit = max(len(a) for a, _ in pairs)
    if not 10 <= limit <= 8192 or not np.isfinite(focal) or focal <= 0:
        raise ValueError("Invalid SimpleVO match count or camera focal length")
    rng = np.random.default_rng(2251)
    match_len = limit * 5
    sample_len = HOMOGRAPHY_TRIALS * SAMPLE_SIZE * 4
    packed = np.zeros((len(pairs), match_len + sample_len + 4), dtype="<f4")
    for index, (a, b) in enumerate(pairs):
        count = len(a)
        if a.shape != (count, 2) or b.shape != a.shape or count < 10:
            raise ValueError(f"SimpleVO pair {index + 1} has fewer than 10 valid correspondences")
        points = np.concatenate(((a - center) / focal, (b - center) / focal), axis=1)
        match = packed[index, :match_len].reshape(limit, 5)
        match[:count, :4] = points
        match[:count, 4] = 1
        samples = [rng.choice(count, SAMPLE_SIZE, replace=False) for _ in range(ESSENTIAL_TRIALS)]
        # The extra homography proposals leave the essential sequence
        # independent of how many planar hypotheses are evaluated.
        homography_rng = deepcopy(rng)
        samples.extend(homography_rng.choice(count, SAMPLE_SIZE, replace=False)
                       for _ in range(HOMOGRAPHY_TRIALS - ESSENTIAL_TRIALS))
        packed[index, match_len:-4] = points[samples].reshape(-1)
        packed[index, -4:] = (tolerance / focal, focal, *center)
    if not np.isfinite(packed).all() or tolerance <= 0:
        raise ValueError("Invalid SimpleVO correspondence data or inlier threshold")
    return packed, limit


def run(probe, backend, keys, pairs, calibration, work, *, repeats=1, expected_device=None, tolerance=4.0):
    schedule = plan(keys)
    frames = len(schedule)
    if len(pairs) != len(keys) - 1:
        raise ValueError("SimpleVO expects one correspondence set per consecutive keyframe pair")
    values, limit = pack(pairs, calibration, tolerance)
    source, interpolation, target = (work / f"simplevo-{kind}.f32" for kind in ("matches", "plan", "output"))
    values.tofile(source)
    schedule.tofile(interpolation)
    cmd = [probe, backend, str(frames), str(len(pairs)), str(limit), source, interpolation, target, str(repeats)]
    result = subprocess.run(cmd, stdout=sys.stderr, stderr=subprocess.PIPE, text=True)
    sys.stderr.write(result.stderr)
    result.check_returncode()
    dispatches = repeats * (len(pairs) + 1)
    verify_placement(result.stderr, backend, dispatches, expected_device)
    output = np.fromfile(target, dtype="<f4")
    offset = len(pairs) * PAIR_WIDTH
    relative = output[:offset].reshape(len(pairs), PAIR_WIDTH)
    trajectory = output[offset:].reshape(frames, TRAJECTORY_WIDTH)
    if not np.isfinite(output).all() or not np.isin(relative[:, 13], (1, 2)).all() or (relative[:, 12] < 10).any():
        raise ValueError("SimpleVO failed to produce valid two-view geometry and a finite trajectory")
    rotation = trajectory[:, :9].reshape(frames, 3, 3)
    validate_rotations(rotation)
    validate_rotations(relative[:, :9].reshape(-1, 3, 3))
    return {"relative": relative, "rotation": np.ascontiguousarray(rotation), "translation": trajectory[:, 9:12].copy(),
            "angular": trajectory[:, 12:].copy(), "dispatches": dispatches}


def validate_rotations(rotations):
    if (not np.isfinite(rotations).all()
            or not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), rtol=0, atol=1e-4)
            or not np.allclose(np.linalg.det(rotations), 1, rtol=0, atol=1e-4)):
        raise ValueError("SimpleVO returned a non-rotation")


def trajectory_reference(keys, relative):
    from scipy.spatial.transform import Rotation, Slerp

    frames = np.arange(keys[-1] + 1)
    r = [np.eye(3)]
    t = [np.zeros(3)]
    for row in relative:
        increment = row[:9].astype(np.float64).reshape(3, 3)
        t.append(increment @ t[-1] + row[9:12])
        r.append(increment @ r[-1])
    rotation = Slerp(keys, Rotation.from_matrix(r))(frames).as_matrix()
    rotation[keys] = r
    translation = np.stack([np.interp(frames, keys, axis) for axis in np.array(t).T], axis=1)
    angular = (rotation[1:] @ rotation[:-1].transpose(0, 2, 1))[:, :2].reshape(-1, 6)
    return {"rotation": rotation, "translation": translation, "angular": np.concatenate((angular, angular[-1:]))}


def upstream_relative(pairs, calibration):
    import pycolmap
    from hmr4d.utils.preproc.relpose.solver_two_view import CameraParams, TwoPairSolver

    # The estimator consumes a thread-local RNG across pairs. Start each
    # reference sequence from the same state, independent of evaluation order.
    pycolmap.set_random_seed(0)
    width, height, focal = calibration
    solver = TwoPairSolver(CameraParams(width, height, focal_length=focal), solver="pycolmap")
    return np.asarray([np.concatenate((pose[:3, :3].reshape(9), pose[:3, 3]))
                       for pose in (solver.solve(a, b) for a, b in pairs)])


def orientation_error(actual, expected):
    from scipy.spatial.transform import Rotation

    relative = np.asarray(actual, dtype=np.float64) @ np.asarray(expected, dtype=np.float64).transpose(0, 2, 1)
    angles = np.rad2deg(Rotation.from_matrix(relative).magnitude())
    return {"max_deg": float(angles.max()), "rms_deg": float(np.sqrt(np.mean(np.square(angles))))}


def compare_trajectory(keys, result):
    expected = trajectory_reference(keys, result["relative"])
    errors = {}
    for key in ("rotation", "translation", "angular"):
        # The upstream quaternion lerp shortcut near 1 and FP32 prefix products
        # can differ slightly from the independent sequential FP64 SLERP.
        np.testing.assert_allclose(result[key], expected[key], rtol=3e-5, atol=3e-5, err_msg=key)
        errors[key] = float(np.abs(result[key] - expected[key]).max())
    return errors
