"""Run fixed- or moving-camera video to SMPL-X with the local CyTools runtime."""

import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from temporal_io import FORK_TOOLS, load_checkpoint, run, save_weights, schema_for
from build import run_environment, scratch_directory
from probe_placement import PLATFORMS
from camera_motion import estimate_rotations, load_pycolmap, max_rotation_degrees
import conditions_io
import decode_io
import postprocess_io
import pose_preprocess
import simplevo_io
import sift_io

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "smplx"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "yolov8"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "botsort"))
import detector_preprocess
import smplx_io


PARTS = ("body_pose", "betas", "global_orient", "transl")
SPACES = ("global", "incam")
NETWORK_OUTPUTS = ("pred_x", "pred_cam", "static_conf_logits")


def use_local_runtime(work):
    os.environ.update(run_environment(work, FORK_TOOLS))
    settings = work / "ultralytics"
    (settings / "Ultralytics").mkdir(parents=True)
    cuda_cache = work / "cuda-cache"
    cuda_cache.mkdir()
    os.environ.update({
        "YOLO_CONFIG_DIR": str(settings),
        "YOLO_AUTOINSTALL": "false",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "MPLCONFIGDIR": str(work / "matplotlib"),
        "WANDB_MODE": "disabled",
        "CUDA_CACHE_PATH": str(cuda_cache),
        "TORCH_EXTENSIONS_DIR": str(work / "torch-extensions"),
        "TRITON_CACHE_DIR": str(work / "triton"),
    })


def prepare(args, work, bridge):
    from fdanyone.video import decode_canonical_clip, write_gvhmr_video
    from hydra import compose, initialize_config_module
    from omegaconf import open_dict
    from hmr4d.utils.geo_transform import compute_cam_angvel
    from hmr4d.utils.video_io_utils import get_video_reader

    clip = decode_canonical_clip(args.video, num_frames=args.frames, start_time=args.start, fps=args.fps)
    working_video = write_gvhmr_video(clip, work / "clip.mp4")
    bridge._verify_gvhmr_decode(clip, working_video, get_video_reader)
    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        cfg = compose(config_name="demo", overrides=[
            "video_name=clip", "static_cam=true", "verbose=false", "use_dpvo=false",
            bridge.hydra_override("output_root", work / "motion"),
        ])
    with open_dict(cfg):
        cfg.video_path = str(working_video)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)
    preprocessing = {"yolo_engine": "pytorch", "vitpose_engine": "pytorch", "hmr2_engine": "pytorch"}
    with (bridge._legacy_checkpoint_loading(),
          detector_preprocess.native_detector(args, work, preprocessing),
          pose_preprocess.native_pose(args, work, preprocessing, clip.rgb_frames)):
        bridge._run_preprocess(cfg)
    data = bridge._load_data(cfg)
    if int(data["length"]) != len(clip.frames):
        raise ValueError("GVHMR video frame count and canonical timeline differ")
    odometry = {}
    if args.camera == "moving":
        if args.odometry_probe:
            if args.sift_probe:
                keys = simplevo_io.keyframe_indices(len(clip.frames))
                keyframes = tuple(clip.rgb_frames[index] for index in keys)
                print(f"SimpleVO ZML: reducing and matching {len(keys)} RGB keyframes", flush=True)
                sift = sift_io.run(args.sift_probe, args.backend, keyframes, work, repeats=2 if args.verify else 1,
                                   expected_device=args.expected_device, full_resolution=True, capture_reduced=args.verify)
                pairs = sift["pairs"]
                calibration = simplevo_io.calibration_for(*sift["image_hw"])
                odometry.update(sift_engine="zml", sift_dispatches=sift["dispatches"],
                                sift_resize_engine="zml", sift_resize_outputs=sift["resize_outputs"],
                                sift_keyframes=len(keys), sift_input_hw=list(keyframes[0].shape[:2]),
                                sift_output_hw=list(sift["image_hw"]),
                                sift_min_matches=min(len(left) for left, _ in pairs),
                                sift_total_matches=sum(len(left) for left, _ in pairs))
                if args.verify:
                    _, images, _ = simplevo_io.keyframes(working_video, len(clip.frames))
                    np.testing.assert_array_equal(sift["images"], images, err_msg="SimpleVO RGB8 keyframes")
                    odometry.update(sift_resize_rgb_values=images.size, sift_resize_max_abs_error=0)
            else:
                keys, pairs, calibration = simplevo_io.collect(working_video, len(clip.frames))
            result = simplevo_io.run(args.odometry_probe, args.backend, keys, pairs, calibration, work,
                                     repeats=2 if args.verify else 1, expected_device=args.expected_device)
            data["camera_rotations_w2c"] = torch.from_numpy(result["rotation"])
            data["camera_translation_unscaled_w2c"] = torch.from_numpy(result["translation"])
            data["cam_angvel"] = torch.from_numpy(result["angular"])
            relative = result["relative"]
            odometry.update(camera_odometry="SimpleVO ZML from SIFT matches", odometry_dispatches=result["dispatches"],
                            odometry_pair_count=len(pairs), odometry_homographies=int(np.count_nonzero(relative[:, 13] == 1)),
                            odometry_min_inliers=int(relative[:, 12].min()))
            if args.verify:
                odometry["odometry_same_pose_max_abs_error"] = simplevo_io.compare_trajectory(keys, result)
                upstream = simplevo_io.upstream_relative(pairs, calibration)
                reference = simplevo_io.trajectory_reference(keys, upstream)
                odometry["odometry_upstream_trajectory_error"] = simplevo_io.orientation_error(result["rotation"], reference["rotation"])
                odometry["odometry_upstream_pair_error"] = simplevo_io.orientation_error(relative[:, :9].reshape(-1, 3, 3),
                                                                                       upstream[:, :9].reshape(-1, 3, 3))
                if args.sift_probe:
                    cv_pairs = simplevo_io.opencv_pairs(images, seed=0)
                    cv_relative = simplevo_io.upstream_relative(cv_pairs, calibration)
                    cv_rotation = simplevo_io.trajectory_reference(keys, cv_relative)["rotation"]
                    odometry["odometry_opencv_pycolmap_trajectory_error"] = simplevo_io.orientation_error(result["rotation"], cv_rotation)
                    odometry["sift_opencv_min_matches"] = min(len(left) for left, _ in cv_pairs)
                    odometry["sift_opencv_total_matches"] = sum(len(left) for left, _ in cv_pairs)
        else:
            rotation = torch.from_numpy(estimate_rotations(working_video, len(clip.frames)))
            data["camera_rotations_w2c"] = rotation
            data["cam_angvel"] = compute_cam_angvel(rotation)
            odometry["camera_odometry"] = "SimpleVO (SIFT/PyCOLMAP)"
        if data["cam_angvel"].shape != (len(clip.frames), 6) or not torch.isfinite(data["cam_angvel"]).all():
            raise ValueError("Invalid GVHMR camera angular velocity")
    data = {key: value.detach().cpu() for key, value in data.items()}
    batch = {key: data[key][None] for key in ("length", "bbx_xys", "K_fullimg", "cam_angvel", "f_imgseq")}
    if args.verify or not args.conditions_probe:
        from hmr4d.utils.geo.hmr_cam import normalize_kp2d

        batch["obs"] = normalize_kp2d(data["kp2d"], data["bbx_xys"])[None]
    return cfg, clip, data, batch, {**preprocessing, **odometry}


def native_conditions(args, pipeline, data, batch, work, metrics, repeats):
    stats = None
    if pipeline.args.normalize_cam_angvel:
        stats = (pipeline.cam_angvel_mean.detach().cpu().numpy(), pipeline.cam_angvel_std.detach().cpu().numpy())
    values = [data[field].numpy() for field in ("kp2d", "bbx_xys", "K_fullimg", "cam_angvel")]
    prepared = conditions_io.run(args.conditions_probe, args.backend, *values, work,
                                 stats=stats, repeats=repeats, expected_device=args.expected_device)
    metrics.update(conditions_engine="zml", conditions_dispatches=repeats)
    if args.verify:
        reference = decode_io.temporal_conditions(pipeline, batch)
        expected = {name: reference[name][0].detach().cpu().numpy() for name in prepared}
        metrics["conditions_max_abs_error"] = conditions_io.compare(prepared, expected)
    output = {name: torch.from_numpy(value.copy())[None] for name, value in prepared.items()}
    batch.setdefault("obs", output["obs"])
    return {**output, "f_imgseq": batch["f_imgseq"]}


class NativeTemporal(torch.nn.Module):
    def __init__(self, probe, backend, weights, work, *, repeats, expected_device, prepared=None):
        super().__init__()
        self.probe, self.backend, self.weights, self.work = probe, backend, weights, work
        self.repeats, self.expected_device = repeats, expected_device
        self.prepared = prepared
        self.called = False

    def forward(self, length, **conditions):
        if self.prepared is not None:
            conditions = self.prepared
        if length.numel() != 1 or int(length.item()) != conditions["obs"].shape[1] or self.called:
            raise ValueError("Expected one unpadded GVHMR clip per video inference")
        self.called = True
        values = {name: value[0].detach().cpu().numpy() for name, value in conditions.items()}
        result = run(self.probe, self.backend, self.weights, values, self.work, "zml-video",
                     repeats=self.repeats, expected_device=self.expected_device)
        return {name: torch.from_numpy(array.copy()).to(conditions["obs"].device)[None] for name, array in result.items()}


def body_parameters(prediction):
    return {
        f"{space}_{name}": prediction[f"pred_smpl_params_{space}"][name][0].detach().cpu().numpy()
        for space in SPACES for name in PARTS
    }


def torch_surfaces(model, params, with_joints=False):
    outputs = {}
    for space in SPACES:
        body = model(**{name: torch.from_numpy(params[f"{space}_{name}"]) for name in PARTS})
        vertices = body.vertices.detach().cpu().numpy()
        frames = len(params[f"{space}_body_pose"])
        if vertices.shape != (frames, smplx_io.VERTICES, 3) or not np.isfinite(vertices).all():
            raise ValueError(f"Invalid SMPL-X vertices in {space} space")
        outputs[f"{space}_vertices"] = vertices
        if with_joints:
            outputs[f"{space}_kinematic_joints"] = body.joints.detach().cpu().numpy()[:, :smplx_io.JOINTS]
    return outputs


def zml_surfaces(args, work, asset, params, repeats):
    weights = work / "smplx.safetensors"
    faces = smplx_io.convert(asset, weights)
    packed = np.concatenate([smplx_io.pack({name: params[f"{space}_{name}"] for name in PARTS}) for space in SPACES])
    vertices, joints, dispatches = smplx_io.run(args.smplx_probe, args.backend, weights, packed, work,
                                              repeats=repeats, expected_device=args.expected_device)
    outputs = {}
    for space, v, j in zip(SPACES, np.split(vertices, 2), np.split(joints, 2)):
        outputs[f"{space}_vertices"] = v
        outputs[f"{space}_kinematic_joints"] = j
    return outputs, faces, dispatches


def apply_corrections(prediction, corrected):
    prediction["pred_smpl_params_global"].update(corrected)
    prediction["pred_smpl_params_incam"]["body_pose"] = corrected["body_pose"]
    prediction["decode_dict"]["body_pose"] = corrected["body_pose"]


@torch.inference_mode()
def recover(args, work, bridge, checkpoint, schema, smplx_asset):
    from hydra.utils import instantiate
    from hmr4d.utils.smplx_utils import make_smplx

    cfg, clip, data, batch, preprocessing = prepare(args, work, bridge)
    print(f"GVHMR video preprocessing: {len(clip.frames)} decoded frames, YOLOv8, ViTPose and HMR2", flush=True)
    state = load_checkpoint(checkpoint, schema)
    weights = work / "temporal.safetensors"
    save_weights(state, weights)
    pipeline = instantiate(cfg.pipeline, _recursive_=False).eval()
    use_zml_smplx = args.smplx_probe is not None
    use_zml_decode = args.decode_probe is not None
    body = make_smplx("supermotion").eval() if args.verify or not use_zml_smplx else None
    static_cam = args.camera == "fixed"
    baseline = baseline_mesh = upstream_mesh = None
    if args.verify:
        pipeline.denoiser3d.load_state_dict(state, strict=True)
        baseline = pipeline(batch, train=False, postproc=True, static_cam=static_cam)
        baseline_mesh = torch_surfaces(body, body_parameters(baseline), with_joints=use_zml_smplx)
        if use_zml_decode:
            upstream_mesh = baseline_mesh
            output = baseline["model_output"]
            baseline = decode_io.assemble(output, decode_io.reference(output, batch, pipeline.endecoder))
            apply_corrections(baseline, postprocess_io.reference(baseline, pipeline.endecoder, args.camera))
            baseline_mesh = torch_surfaces(body, body_parameters(baseline), with_joints=use_zml_smplx)
    repeats = 2 if args.verify else 1
    prepared = None
    if args.conditions_probe:
        prepared = native_conditions(args, pipeline, data, batch, work, preprocessing, repeats)
    native = NativeTemporal(args.probe, args.backend, weights, work, repeats=repeats, expected_device=args.expected_device,
                            prepared=prepared)
    pipeline.denoiser3d = native
    use_zml_postprocess = args.postprocess_probe is not None
    decode_summary = {"decode_engine": "zml" if use_zml_decode else "pytorch"}
    if use_zml_decode:
        conditions = prepared if prepared is not None else decode_io.temporal_conditions(pipeline, batch)
        output = native(batch["length"], **conditions)
        stats = work / "decode-stats.safetensors"
        decode_io.save_stats(pipeline.endecoder, stats)
        decoded = decode_io.run(args.decode_probe, args.backend, stats, output, batch, work,
                                repeats=repeats, expected_device=args.expected_device)
        prediction = decode_io.assemble(output, decoded)
        decode_summary["decode_inferences"] = repeats
        if args.verify:
            expected = decode_io.reference(output, batch, pipeline.endecoder)
            decode_summary["decode_same_input_max_abs_error"] = decode_io.compare(decoded, expected)
            upstream = decode_io.reference(output, batch, pipeline.endecoder, stable=False)
            decode_summary["decode_upstream_max_abs_error"] = decode_io.compare(decoded, upstream, verify=False)
    else:
        prediction = pipeline(batch, train=False, postproc=not use_zml_postprocess, static_cam=static_cam)
    if not native.called:
        raise RuntimeError("The pipeline did not invoke the ZML temporal model")
    post_summary = {"postprocess_engine": "zml" if use_zml_postprocess else "pytorch"}
    if use_zml_postprocess:
        skeleton = work / "postprocess-skeleton.safetensors"
        postprocess_io.save_skeleton(pipeline.endecoder, skeleton)
        corrected = postprocess_io.run(args.postprocess_probe, args.backend, args.camera, skeleton, prediction, work,
                                       repeats=repeats, expected_device=args.expected_device)
        post_summary["postprocess_inferences"] = repeats
        if args.verify:
            expected = postprocess_io.reference(prediction, pipeline.endecoder, args.camera)
            post_summary["postprocess_same_input_max_abs_error"] = postprocess_io.compare(corrected, expected)
            original = prediction["pred_smpl_params_global"]
            post_summary["postprocess_max_abs_update"] = {
                key: float((value - original[key]).abs().max()) for key, value in corrected.items()
            }
    elif use_zml_decode:
        corrected = postprocess_io.reference(prediction, pipeline.endecoder, args.camera)
    if use_zml_postprocess or use_zml_decode:
        apply_corrections(prediction, corrected)
    params = body_parameters(prediction)
    if use_zml_smplx:
        mesh_outputs, faces, mesh_dispatches = zml_surfaces(args, work, smplx_asset, params, repeats)
    else:
        mesh_outputs = torch_surfaces(body, params)
        faces = np.asarray(body.faces, dtype=np.int32)
    if faces.shape != (smplx_io.TRIANGLES, 3) or faces.min() < 0 or faces.max() >= smplx_io.VERTICES:
        raise ValueError("Invalid SMPL-X mesh connectivity")
    meshes = {**params, **mesh_outputs}
    timestamps = np.asarray([float(frame.canonical_timestamp) for frame in clip.frames])
    source_frames = np.asarray([frame.source_index for frame in clip.frames])
    summary = {"frames": len(clip.frames), "unique_source_frames": len(np.unique(source_frames)),
               "fps": float(clip.fps), "vertices_per_frame": smplx_io.VERTICES, "triangles": len(faces),
               "camera_mode": args.camera, "zml_backend": args.backend,
               "zml_pjrt": PLATFORMS[args.backend], "zml_inferences": repeats,
               "smplx_engine": "zml" if use_zml_smplx else "pytorch", **decode_summary, **post_summary, **preprocessing}
    if use_zml_smplx:
        summary.update(smplx_dispatches=mesh_dispatches, smplx_kinematic_joints=smplx_io.JOINTS)
    camera_fields = {}
    if not static_cam:
        rotation = data["camera_rotations_w2c"].numpy()
        camera_fields = {"camera_rotations_w2c": rotation, "camera_angular_velocity_6d": data["cam_angvel"].numpy()}
        if "camera_translation_unscaled_w2c" in data:
            camera_fields["camera_translation_unscaled_w2c"] = data["camera_translation_unscaled_w2c"].numpy()
        summary["max_camera_rotation_deg"] = max_rotation_degrees(rotation)
    if args.verify:
        errors = {}
        for key in NETWORK_OUTPUTS:
            actual = prediction["model_output"][key].detach().cpu().numpy()
            expected = baseline["model_output"][key].detach().cpu().numpy()
            np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4, equal_nan=False)
            errors[key] = float(np.abs(actual - expected).max())
        for key, value in mesh_outputs.items():
            np.testing.assert_allclose(value, baseline_mesh[key], atol=8e-4, rtol=2e-4, equal_nan=False)
            errors[key] = float(np.abs(value - baseline_mesh[key]).max())
        summary["max_abs_error"] = errors
        if upstream_mesh is not None:
            summary["upstream_mesh_max_abs_error"] = {
                key: float(np.abs(value - upstream_mesh[key]).max()) for key, value in mesh_outputs.items()
            }
        if use_zml_smplx:
            np.testing.assert_array_equal(faces, body.faces)
            same_input = torch_surfaces(body, params, with_joints=True)
            errors = {}
            for key, value in mesh_outputs.items():
                np.testing.assert_allclose(value, same_input[key], atol=3e-5, rtol=2e-5, equal_nan=False)
                errors[key] = float(np.abs(value - same_input[key]).max())
            summary["smplx_same_input_max_abs_error"] = errors
    destination = args.output or work / "verified-smplx.npz"
    with destination.open("xb") as stream:
        np.savez(
            stream, **meshes, **camera_fields, faces=faces, fps=np.array(float(clip.fps)), timestamps=timestamps,
            source_frame_indices=source_frames,
            intrinsics=data["K_fullimg"].numpy(), observed_keypoints_2d=data["kp2d"].numpy(),
        )
    if args.verify:
        with np.load(destination, allow_pickle=False) as exported:
            np.testing.assert_array_equal(exported["timestamps"], timestamps)
            np.testing.assert_array_equal(exported["source_frame_indices"], source_frames)
            np.testing.assert_array_equal(exported["faces"], faces)
            for key, value in {**meshes, **camera_fields}.items():
                np.testing.assert_array_equal(exported[key], value)
    if args.output:
        summary["output"] = str(args.output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("probe", "four-d-anyone", "upstream", "video"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--smplx-probe", type=Path, help="native SMPL-X executable for mesh and joint evaluation")
    parser.add_argument("--decode-probe", type=Path, help="native GVHMR body and root parameter decoder")
    parser.add_argument("--conditions-probe", type=Path, help="native GVHMR keypoint and camera conditions executable")
    parser.add_argument("--odometry-probe", type=Path, help="native SimpleVO two-view geometry and trajectory executable")
    parser.add_argument("--sift-probe", type=Path, help="native SimpleVO half-scale, SIFT descriptor extraction and matching executable")
    parser.add_argument("--postprocess-probe", type=Path, help="native GVHMR contact and inverse kinematics executable")
    parser.add_argument("--vitpose-probe", type=Path, help="native ViTPose-H heatmap executable")
    parser.add_argument("--hmr2-probe", type=Path, help="native HMR2 appearance token executable")
    parser.add_argument("--crop-probe", type=Path, help="shared native RGB half-scale, crop and normalization for ViTPose and HMR2")
    parser.add_argument("--yolov8-probe", type=Path, help="native YOLOv8x COCO detector executable")
    parser.add_argument("--botsort-cost-probe", type=Path, help="native BoT-SORT association costs with --yolov8-probe")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--fps")
    parser.add_argument("--camera", choices=("fixed", "moving"), default="fixed")
    parser.add_argument("--pycolmap-site", type=Path, help="local site-packages directory containing PyCOLMAP for moving cameras")
    parser.add_argument("--backend", choices=PLATFORMS, default="cpu")
    parser.add_argument("--expected-device", help="substring of the device that must own each native output")
    args = parser.parse_args()
    if not 1 <= args.frames <= 4096 or (args.output is None and not args.verify):
        parser.error("Supply 1–4096 frames and either --output or --verify")
    if args.camera == "moving" and args.frames < 2:
        parser.error("Moving-camera inference needs at least two frames")
    if args.odometry_probe and args.camera != "moving":
        parser.error("--odometry-probe requires --camera moving")
    if args.sift_probe and not args.odometry_probe:
        parser.error("--sift-probe requires --odometry-probe")
    if args.botsort_cost_probe and not args.yolov8_probe:
        parser.error("--botsort-cost-probe requires --yolov8-probe")
    if args.crop_probe and not (args.vitpose_probe or args.hmr2_probe):
        parser.error("--crop-probe requires a native ViTPose or HMR2 probe")
    if args.output and (args.output.exists() or args.output.suffix.lower() != ".npz"):
        parser.error("--output must be a new .npz file")
    args.video = args.video.resolve(strict=True)
    args.probe = args.probe.resolve(strict=True)
    if args.smplx_probe:
        args.smplx_probe = args.smplx_probe.resolve(strict=True)
    if args.decode_probe:
        args.decode_probe = args.decode_probe.resolve(strict=True)
    if args.conditions_probe:
        args.conditions_probe = args.conditions_probe.resolve(strict=True)
    if args.odometry_probe:
        args.odometry_probe = args.odometry_probe.resolve(strict=True)
    if args.sift_probe:
        args.sift_probe = args.sift_probe.resolve(strict=True)
    if args.postprocess_probe:
        args.postprocess_probe = args.postprocess_probe.resolve(strict=True)
    if args.vitpose_probe:
        args.vitpose_probe = args.vitpose_probe.resolve(strict=True)
    if args.hmr2_probe:
        args.hmr2_probe = args.hmr2_probe.resolve(strict=True)
    if args.crop_probe:
        args.crop_probe = args.crop_probe.resolve(strict=True)
    if args.yolov8_probe:
        args.yolov8_probe = args.yolov8_probe.resolve(strict=True)
    if args.botsort_cost_probe:
        args.botsort_cost_probe = args.botsort_cost_probe.resolve(strict=True)
    args.output = args.output.resolve() if args.output else None
    sys.path.insert(0, str(args.four_d_anyone.resolve(strict=True)))
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    with redirect_stdout(sys.stderr), scratch_directory() as work:
        use_local_runtime(work)
        if not torch.cuda.is_available():
            raise SystemExit("The local CyTools video feature extractors require CUDA")
        pycolmap = load_pycolmap(args.pycolmap_site) if args.camera == "moving" and (args.verify or not args.odometry_probe) else None
        from fdanyone.motion import gvhmr as bridge

        upstream, revision = bridge.validate_gvhmr(args.upstream)
        with bridge.gvhmr_imports(upstream):
            bridge.install_pytorch3d_compat()
            bridge._register_inference_store()
            summary = recover(args, work, bridge, upstream / bridge.GVHMR_ASSETS[0], schema_for(args.probe),
                              upstream / smplx_io.ASSET)
        summary["gvhmr_revision"] = revision
        summary["preprocessing_device"] = torch.cuda.get_device_name()
        if pycolmap is not None:
            summary["pycolmap_version"] = pycolmap.__version__
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
