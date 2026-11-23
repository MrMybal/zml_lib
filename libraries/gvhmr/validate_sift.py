"""Validate ZML SIFT, half-scale and SimpleVO on synthetic data or a local video."""

import argparse
from contextlib import redirect_stdout
from itertools import islice
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build import ROOT, run_environment, scratch_directory
from probe_placement import PLATFORMS
import sift_io
import simplevo_io


FORK_TOOLS = ROOT.parent / "zml_win" / "bazel-win-tools"


def use_scratch(work):
    cuda_cache = work / "cuda-cache"
    cuda_cache.mkdir()
    os.environ.update(run_environment(work, FORK_TOOLS), CUDA_CACHE_PATH=str(cuda_cache))


def fixture():
    import cv2

    rng = np.random.default_rng(952)
    height, width = 256, 320
    image = cv2.GaussianBlur(rng.integers(15, 105, (height, width), dtype=np.uint8), (0, 0), 1.2)
    for _ in range(240):
        x, y = rng.integers((12, 12), (width - 12, height - 12)).tolist()
        cv2.circle(image, (x, y), int(rng.integers(2, 11)), int(rng.integers(35, 255)), int(rng.choice([-1, 1, 2])),
                   cv2.LINE_AA)
        dx, dy = rng.integers(-10, 11, 2).tolist()
        cv2.line(image, (x, y), (x + dx, y + dy), int(rng.integers(30, 235)), 1, cv2.LINE_AA)
    shift = np.array([[1., 0., 9.5], [0., 1., -5.75]])
    rotation = cv2.getRotationMatrix2D((width / 2, height / 2), 18.0, 1.09)
    images = np.stack([image, *[cv2.warpAffine(image, transform, (width, height), borderValue=48)
                                for transform in (shift, rotation)], np.full_like(image, 48)])
    relative_rotation = rotation @ np.vstack((cv2.invertAffineTransform(shift), [0, 0, 1]))
    return np.repeat(images[:, :, :, None], 3, axis=-1), (shift, relative_rotation)


def geometric_error(pairs, transforms):
    output = []
    for (left, right), transform in zip(pairs, transforms):
        expected = left @ transform[:, :2].T + transform[:, 2]
        error = np.linalg.norm(expected - right, axis=1)
        output.append({"matches": len(left), "inlier_fraction_2px": float(np.mean(error < 2)),
                       "median_px": float(np.median(error))})
    return output


def validate_matching(features, raw):
    accepted = 0
    for left, right, observed in zip(features[:-1], features[1:], raw):
        left = left[left[:, 2] == 1]
        right = right[right[:, 2] == 1]
        if len(right) < 2:
            if observed[:, 4].any():
                raise ValueError("Accepted a match without a second descriptor")
            continue
        found = observed[:len(left), 4] == 1
        dist = cdist(left[:, 3:], right[:, 3:], "sqeuclidean")
        neighbors = np.argsort(dist, axis=1, kind="stable")[:, :2]
        top = np.take_along_axis(dist, neighbors, axis=1)
        margin = 0.49 * top[:, 1] - top[:, 0]
        stable = np.abs(margin) > 2e-5
        np.testing.assert_array_equal(found[stable], margin[stable] > 0)
        if observed[len(left):, 4].any():
            raise ValueError("Accepted a padded left descriptor")
        np.testing.assert_array_equal(observed[:len(left), :2], left[:, :2])
        np.testing.assert_allclose(observed[:len(left), 2:4][found], right[neighbors[:, 0], :2][found], atol=1e-5, rtol=0)
        accepted += int(found.sum())
    return accepted


def local_opencv_pairs(images):
    import cv2

    sift = cv2.SIFT_create()
    features = [sift.detectAndCompute(image[:, :, 0], None) for image in images[:3]]
    pairs = []
    for (left, a), (right, b) in zip(features[:-1], features[1:]):
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        good = [(left[m.queryIdx].pt, right[m.trainIdx].pt) for m, n in matcher.knnMatch(a, b, k=2)
                if m.distance < 0.7 * n.distance]
        pairs.append(tuple(np.asarray(good, dtype=np.float32).transpose(1, 0, 2)))
    return pairs


def validate(args, work):
    images, transforms = fixture()
    result = sift_io.run(args.probe, args.backend, images, work, descriptors=True, repeats=2, expected_device=args.expected_device)
    features = result["descriptors"]
    if not np.isfinite(features).all() or not np.isin(features[:, :, 2], (0, 1)).all():
        raise ValueError("Invalid SIFT descriptors")
    counts = features[:, :, 2].sum(axis=1).astype(int)
    if any(count < 75 for count in counts[:3]) or counts[-1] != 0:
        raise ValueError(f"Unexpected feature distribution {counts}")
    vectors = features[:, :, 3:][features[:, :, 2] == 1]
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-5, rtol=0)
    if (vectors < 0).any():
        raise ValueError("Negative SIFT histogram bins")
    total = validate_matching(features, result["raw_matches"])
    native = geometric_error(result["pairs"], transforms)
    if any(pair["matches"] < 30 or pair["inlier_fraction_2px"] < 0.9 or pair["median_px"] > 0.8 for pair in native):
        raise ValueError(f"SIFT invariance check failed: {native}")
    return {"descriptors_per_frame": counts.tolist(), "exact_matching_matches": total, "sift_dispatches": result["dispatches"],
            "zml": native, "opencv": geometric_error(local_opencv_pairs(images), transforms)}


def compare_resize(args, frames, work, case):
    from validate_half import reference_half

    expected = reference_half(frames)
    options = {"expected_device": args.expected_device, "descriptors": True}
    native_dir, reference_dir = work / "native", work / "reference"
    native_dir.mkdir()
    reference_dir.mkdir()
    native = sift_io.run(args.probe, args.backend, frames, native_dir,
                         full_resolution=True, capture_reduced=True, repeats=2, **options)
    np.testing.assert_array_equal(native["images"], expected, err_msg=case)
    reference = sift_io.run(args.probe, args.backend, expected, reference_dir, **options)
    for field in ("descriptors", "raw_matches"):
        np.testing.assert_array_equal(native[field], reference[field], err_msg=f"{case} {field}")
    counts = [len(left) for left, _ in native["pairs"]]
    summary = {"frames": len(frames), "input_hw": list(frames[0].shape[:2]),
               "output_hw": list(native["image_hw"]), "rgb_values": expected.size,
               "resize_outputs": native["resize_outputs"], "sift_dispatches": native["dispatches"],
               "max_abs_rgb_error": 0, "descriptors_identical": True, "matches_identical": True,
               "matches_per_pair": counts}
    return native["pairs"], summary


def validate_resize(args):
    gray = fixture()[0][..., 0]
    color = np.stack((gray, np.roll(gray, 3, axis=2), 255 - gray // 2), axis=-1)
    results = {}
    for key, frames in (("even", color), ("odd", color[:, :-1, :-1].copy())):
        with scratch_directory() as work:
            use_scratch(work)
            pairs, results[key] = compare_resize(args, tuple(frames), work, key)
            counts = results[key]["matches_per_pair"]
            if min(counts[:2]) == 0 or counts[-1] != 0:
                raise AssertionError(f"Unexpected match counts for {key}: {counts}")
            if key == "odd":
                production = work / "production"
                production.mkdir()
                result = sift_io.run(args.probe, args.backend, tuple(frames), production, full_resolution=True,
                                     expected_device=args.expected_device)
                for actual, target in zip(result["pairs"], pairs, strict=True):
                    np.testing.assert_array_equal(actual, target, err_msg="without RGB readback")
                results[key]["matches_identical_without_readback"] = True
    return results


def validate_video(args, work):
    import av

    keys = simplevo_io.keyframe_indices(args.frames)
    selected = set(keys.tolist())
    with av.open(str(args.video)) as source:
        if not source.streams.video:
            raise ValueError(f"No video stream in {args.video}")
        rate = source.streams.video[0].average_rate
        frames = tuple(frame.to_ndarray(format="rgb24")
                       for index, frame in enumerate(islice(source.decode(video=0), args.frames))
                       if index in selected)
    if len(frames) != len(keys):
        raise ValueError(f"The clip has fewer than {args.frames} decoded frames")
    print(f"SimpleVO ZML: comparing {len(keys)} keyframes from {args.frames} decoded frames", flush=True)
    pairs, summary = compare_resize(args, frames, work, args.video.name)
    calibration = simplevo_io.calibration_for(*summary["output_hw"])
    print("SimpleVO ZML: verifying two-view geometry and trajectory", flush=True)
    trajectory = simplevo_io.run(args.odometry_probe, args.backend, keys, pairs, calibration, work,
                                 repeats=2, expected_device=args.expected_device)
    summary.update(frames=args.frames, keyframes=len(keys), fps=float(rate) if rate else None,
                   pyav=av.__version__, libswscale=av.library_versions["libswscale"],
                   matches=sum(summary["matches_per_pair"]), pairs=len(pairs),
                   homographies=int(np.count_nonzero(trajectory["relative"][:, 13] == 1)),
                   min_inliers=int(trajectory["relative"][:, 12].min()),
                   odometry_dispatches=trajectory["dispatches"],
                   trajectory_max_abs_error=simplevo_io.compare_trajectory(keys, trajectory))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--backend", required=True, choices=PLATFORMS)
    parser.add_argument("--expected-device")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--half", action="store_true", help="compare native half-scale and SIFT with PyAV on synthetic RGB images")
    mode.add_argument("--video", type=Path, help="validate half-scale, SIFT and SimpleVO on a local video")
    parser.add_argument("--frames", type=int, help="number of decoded video frames to check, default 195")
    parser.add_argument("--odometry-probe", type=Path, help="native SimpleVO executable for --video")
    args = parser.parse_args()
    if args.video:
        args.frames = 195 if args.frames is None else args.frames
        if not args.odometry_probe or not 2 <= args.frames <= 4096:
            parser.error("--video requires --odometry-probe and a frame count from 2 to 4096")
        args.video = args.video.resolve(strict=True)
        args.odometry_probe = args.odometry_probe.resolve(strict=True)
    elif args.odometry_probe or args.frames is not None:
        parser.error("--frames and --odometry-probe require --video")
    args.probe = args.probe.resolve(strict=True)
    with redirect_stdout(sys.stderr):
        if args.half:
            summary = validate_resize(args)
        else:
            with scratch_directory() as work:
                use_scratch(work)
                summary = validate_video(args, work) if args.video else validate(args, work)
    print(json.dumps(summary, allow_nan=False))


if __name__ == "__main__":
    main()
