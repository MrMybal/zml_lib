"""Compare native Qwen Image Edit with the local ComfyUI PyTorch source."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import NamedTuple

import numpy as np

from inputs import unpack_latent, write_denoise_input
from pipeline import edit_prompt, encode_conditioning, run, run_denoiser, vae_run

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.build import scratch_directory
from scripts.probe_placement import PLATFORMS


def report_delta(label, actual, expected, **metadata):
    delta = np.abs(actual - expected)
    rmse = float(np.sqrt(np.mean(delta**2)))
    print(json.dumps({"stage": label, "shape": list(actual.shape), "max_abs": float(delta.max()),
                      "rmse": rmse, **metadata}, allow_nan=False), flush=True)
    return rmse


def compare(label, actual, expected, placement, atol, rtol, *, rmse_limit=None, elementwise=True):
    if actual.shape != expected.shape:
        raise ValueError(f"{label}: expected shape {expected.shape}, received {actual.shape}")
    rmse = report_delta(label, actual, expected, placement=placement)
    if elementwise:
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False)
    if rmse_limit is not None and rmse > rmse_limit:
        raise AssertionError(f"{label}: RMSE {rmse:.9g} exceeds {rmse_limit:.9g}")


def image_tensor(value):
    import torch

    return torch.from_numpy(np.array(value, dtype=np.float32, copy=True))[None, :, None]


def text_tensor(value):
    import torch

    return torch.from_numpy(np.array(value, dtype=np.float32, copy=True))[None]


def image_model(*, fixture=False):
    import torch
    from comfy.ldm.qwen_image.model import QwenImageTransformer2DModel

    return QwenImageTransformer2DModel(
        num_layers=2 if fixture else 60, joint_attention_dim=32 if fixture else 3584,
        num_attention_heads=3 if fixture else 24, attention_head_dim=16 if fixture else 128,
        axes_dims_rope=(4, 6, 6) if fixture else (16, 56, 56), operations=torch.nn,
        default_ref_method="index_timestep_zero", dtype=torch.float32,
    ).eval()


class ReferenceTrajectory(NamedTuple):
    final: np.ndarray
    later_targets: list[np.ndarray]
    velocities: list[np.ndarray]


def torch_denoiser(model, target, references, text, mask, times, *, capture_trajectory=False, block_reference=None):
    import torch

    device = next(model.parameters()).device
    x = image_tensor(target).to(device)
    context = text_tensor(text).to(device)
    mask = torch.tensor(mask[None], dtype=torch.int64, device=device)
    ref_latents = [image_tensor(value).to(device) for value in references]

    def predict(time):
        return model._forward(x, torch.tensor([time], dtype=torch.float32, device=device), context, attention_mask=mask,
                              ref_latents=ref_latents, transformer_options={})

    if len(times) == 1:
        if capture_trajectory:
            raise ValueError("An Euler trajectory needs a starting and an ending time")
        return predict(times[0])[0, :, 0].cpu().numpy()
    later_targets, velocities = [], []
    for i, time in enumerate(times[:-1]):
        if i == 0 and block_reference is not None:
            with block_reference.capture(model):
                velocity = predict(time)
        else:
            velocity = predict(time)
        if capture_trajectory:
            if i:
                later_targets.append(x[0, :, 0].cpu().numpy())
            velocities.append(velocity[0, :, 0].cpu().numpy())
        x = x + float(times[i + 1] - time) * velocity
    final = x[0, :, 0].cpu().numpy()
    return ReferenceTrajectory(final, later_targets, velocities) if capture_trajectory else final


def report_trajectory(trajectory, target, times, reference, device, placement):
    final, *velocities = trajectory
    fixed_inputs_final = target.copy()
    reference_integrated = target.copy()
    metadata = {"reference_device": device, "placement": placement}
    for i, (actual, expected) in enumerate(zip(velocities, reference.velocities, strict=True)):
        report_delta("Same-input denoiser velocity", actual, expected, step=i + 1, time=float(times[i]), **metadata)
        dt = times[i + 1] - times[i]
        fixed_inputs_final += dt * actual
        reference_integrated += dt * expected
    report_delta("Euler arithmetic with PyTorch velocities", reference_integrated, reference.final, reference_device=device)
    report_delta("Euler drift with reference inputs", fixed_inputs_final, reference.final, **metadata)
    report_delta("Effect of feeding back ZML latents", final, fixed_inputs_final, **metadata)


def run_feedback(args, work, profile, weights, target, references, text, mask, times, baseline, *, sweep=True):
    source, output = work / "edit-euler.input.f32", work / "feedback.output.f32"
    geometry = write_denoise_input(source, target, references, text, mask, times,
                                   reference_states=baseline.later_targets)
    fractions = (0.25, 0.5, 0.75) if sweep else ()
    flag = "--reference-feedback" if sweep else "--reference-feedback-endpoints"
    placement = run(args, args.probe, profile, weights, geometry, source, output, 3 + len(fractions), flags=(flag,))
    rows = np.fromfile(output, dtype="<f4").reshape(5 + 2 * len(fractions), target.size)
    final, first, fed_back, second, forced, *interpolated = (
        unpack_latent(row, *target.shape[1:]) for row in rows
    )
    samples = [(0.0, baseline.later_targets[0], forced)]
    samples.extend(zip(fractions, interpolated[::2], interpolated[1::2], strict=True))
    samples.append((1.0, fed_back, second))
    report_delta("Native first Euler update versus host replay", fed_back, target + (times[1] - times[0]) * first,
                 placement=placement)
    report_delta("Native second Euler update versus host replay", final, fed_back + (times[2] - times[1]) * second,
                 placement=placement)
    return [final, first, forced], placement, samples


def report_feedback(samples, baseline, times, placement, reference_velocity):
    original = baseline.later_targets[0]
    reference_on_native_latent = None
    for fraction, state, velocity in samples:
        expected = baseline.velocities[1] if fraction == 0 else reference_velocity(state)
        report_delta("Intermediate latent displacement", state, original, fraction=fraction)
        report_delta("PyTorch second-step response", expected, baseline.velocities[1], fraction=fraction)
        report_delta("Second-step velocity on the same latent", velocity, expected, fraction=fraction, placement=placement)
        final = state + (times[2] - times[1]) * expected
        report_delta("Final latent with PyTorch second velocity", final, baseline.final, fraction=fraction)
        if fraction == 1:
            reference_on_native_latent = expected
    if reference_on_native_latent is None:
        raise ValueError("Feedback diagnostics require the second-step input produced by ZML")
    return reference_on_native_latent


def fixture(args, work):
    import torch
    from block_diagnostics import BlockReference, compare_blocks
    from prepare import schema_for
    from safetensors.torch import save_file

    torch.manual_seed(94)
    reference = image_model(fixture=True)
    state = {name: value.detach().contiguous() for name, value in reference.state_dict().items() if value.numel()}
    schema = schema_for(args.probe, "fixture")
    if schema != {name: tuple(value.shape) for name, value in state.items()}:
        raise AssertionError("Qwen Image fixture schema differs from ComfyUI")
    rng = np.random.default_rng(63)
    target = rng.standard_normal((16, 4, 6), dtype=np.float32)
    refs = [rng.standard_normal((16, 4, 4), dtype=np.float32), rng.standard_normal((16, 2, 4), dtype=np.float32)]
    text = rng.standard_normal((7, 32), dtype=np.float32)
    mask = np.array([1, 1, 0, 0, 1, 1, 0])
    weights = work / "fixture.safetensors"
    save_file({**state, "weight_scales": torch.ones(len(schema))}, weights)
    schedule = np.array([0.63], dtype="<f4")
    actual, device = run_denoiser(args, work, "fixture", weights, target, [], text, mask, schedule, "velocity")
    expected = torch_denoiser(reference, target, [], text, mask, schedule)
    compare("Fixture FP32 velocity", actual, expected, device, 5e-5, 5e-5)

    scales = torch.ones(len(schema))
    stored, decoded = dict(state), dict(state)
    for index, (name, shape) in enumerate(schema.items()):
        if name.endswith(".weight") and len(shape) == 2:
            scales[index] = state[name].abs().max() / 224
            codes = (state[name] / scales[index]).to(torch.float8_e4m3fn)
            stored[name] = codes
            decoded[name] = codes.float() * scales[index]
    weights = work / "fixture-e4m3.safetensors"
    save_file({**stored, "weight_scales": scales}, weights)
    # The ComfyUI marker is an empty buffer and does not belong to the native schema.
    reference.load_state_dict({**decoded, "__index_timestep_zero__": reference.__index_timestep_zero__})
    schedule = np.array([0.9, 0.55, 0.0], dtype="<f4")
    actual, device = run_denoiser(args, work, "fixture", weights, target, refs, text, mask, schedule, "euler")
    expected = torch_denoiser(reference, target, refs, text, mask, schedule)
    compare("Fixture E4M3FN Euler with two references", actual, expected, device, 5e-5, 5e-5)

    block_reference = BlockReference()
    recorded = torch_denoiser(reference, target, refs, text, mask, schedule, capture_trajectory=True,
                              block_reference=block_reference)
    forced_target = recorded.later_targets[0] + np.float32(0.25)
    trace, device = run_denoiser(args, work, "fixture", weights, target, refs, text, mask, schedule, "trajectory",
                                 reference_states=[forced_target])
    np.testing.assert_array_equal(trace[0], actual)
    compare("Fixture shared first velocity", trace[1], recorded.velocities[0], device, 5e-5, 5e-5)
    expected_forced = torch_denoiser(reference, forced_target, refs, text, mask, schedule[1:2])
    compare("Fixture forced second velocity", trace[2], expected_forced, device, 5e-5, 5e-5)
    comparison = ReferenceTrajectory(recorded.final, [forced_target], [recorded.velocities[0], expected_forced])
    feedback, placement, samples = run_feedback(args, work, "fixture", weights, target, refs, text, mask,
                                                 schedule, comparison)
    np.testing.assert_array_equal(np.stack(feedback), np.stack(trace))
    for fraction, sample, velocity in samples:
        expected_velocity = torch_denoiser(reference, sample, refs, text, mask, schedule[1:2])
        compare(f"Fixture second velocity on shared latent ({fraction})", velocity, expected_velocity, placement, 5e-5, 5e-5)
    compare_blocks(args, run, work, "fixture", weights, target, refs, text, mask, schedule[:1], block_reference,
                   recorded.velocities[0], trace[1], tolerance=5e-5)
    prefix = work / "prefix.output.f32"
    geometry = (target.size // 64, (target.size + sum(value.size for value in refs)) // 64, len(text), 0)
    placement = run(args, args.probe, "fixture", weights, geometry, work / "blocks.input.f32", prefix, 1,
                    flags=("--reference-prefix=1",))
    prefix_velocity = unpack_latent(np.fromfile(prefix, "<f4"), *target.shape[1:])
    compare("Fixture velocity after replacing the first block's output", prefix_velocity,
            recorded.velocities[0], placement, 5e-5, 5e-5)


def reference_vae(source):
    import torch
    from comfy.ldm.wan.vae import WanVAE
    from safetensors.torch import load_file

    with torch.device("meta"):
        model = WanVAE(dim=96, z_dim=16, dim_mult=[1, 2, 4, 4], temperal_downsample=[False, True, True]).eval()
    model.load_state_dict({key: tensor.float() for key, tensor in load_file(source).items()}, strict=True, assign=True)
    return model


def conditioning(args, work, image):
    vision_size = (args.vision_size, args.vision_size) if args.vision_size else (84, 56)
    prompt = edit_prompt("Replace the orange shape with a purple square.")
    return encode_conditioning(args, work, image, vision_size, prompt)


def torch_full_denoiser(source, schema, target, references, text, mask, times, device="cpu", *, capture_trajectory=False,
                        block_reference=None):
    import torch
    from safetensors import safe_open

    with torch.device("meta"):
        model = image_model()
    expected_names = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items() if tensor.numel()}
    if expected_names != schema:
        raise AssertionError("The local ComfyUI transformer does not match the native Edit 2511 schema")
    with safe_open(source, framework="pt", device="cpu") as checkpoint:
        def value(key):
            tensor = checkpoint.get_tensor(key)
            if tensor.dtype == torch.float8_e4m3fn:
                scale = checkpoint.get_tensor(key.removesuffix(".weight") + ".weight_scale").item()
                return (tensor.float() * scale).to(device)
            return tensor.float().to(device)

        top = {key: value(key) for key in schema if not key.startswith("transformer_blocks.")}
        loaded = model.load_state_dict(top, strict=False, assign=True)
        if loaded.unexpected_keys or any(not key.startswith("transformer_blocks.") and key != "__index_timestep_zero__"
                                        for key in loaded.missing_keys):
            raise AssertionError("Failed to load the top-level transformer parameters")

        def hooks(layer, index):
            names = tuple(layer.state_dict())

            def before(module, _):
                weights = {key: value(f"transformer_blocks.{index}.{key}") for key in names}
                module.load_state_dict(weights, strict=True, assign=True)

            def after(module, _, __):
                module.to("meta")

            return layer.register_forward_pre_hook(before), layer.register_forward_hook(after)

        handles = [handle for index, layer in enumerate(model.transformer_blocks) for handle in hooks(layer, index)]
        try:
            return torch_denoiser(model, target, references, text, mask, times, capture_trajectory=capture_trajectory,
                                  block_reference=block_reference)
        finally:
            for handle in handles:
                handle.remove()


def full(args, work):
    import torch
    from PIL import Image, ImageDraw
    from comfy.latent_formats import Wan21
    from prepare import prepare, schema_for

    width, height = args.width, args.height
    h, w = height // 8, width // 8
    devices = args.reference_device or ("cpu",)
    vae_device = devices[0]
    image = Image.new("RGB", (width, height), (21, 42, 85))
    draw = ImageDraw.Draw(image)
    draw.ellipse((11 * width // 64, 8 * height // 48, 43 * width // 64, 36 * height // 48), fill=(229, 141, 48))
    draw.rectangle((29 * width // 64, 34 * height // 48, 58 * width // 64, 43 * height // 48), fill=(38, 203, 155))
    rgb = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / np.float32(127.5) - np.float32(1)
    reference = reference_vae(args.vae)
    normalizer = Wan21()
    encode_path, decode_path = work / "encode.safetensors", work / "decode.safetensors"
    for name, target in (("vae_encode", encode_path), ("vae_decode", decode_path)):
        prepare(args.vae, target, schema_for(args.probe, name), name)
    latent, placement = vae_run(args, work, "vae_encode", encode_path, rgb, h, w, "encode")
    decoded, placement = vae_run(args, work, "vae_decode", decode_path, latent, h, w, "decode")
    reference.to(vae_device)
    expected = normalizer.process_in(reference.encode(image_tensor(rgb).to(vae_device)))[0, :, 0].cpu().numpy()
    compare(f"VAE normalized encoder (PyTorch {vae_device})", latent, expected, placement, 8e-4, 6e-4)
    expected = reference.decode(normalizer.process_out(image_tensor(latent).to(vae_device))).clamp(-1, 1)[0, :, 0].cpu().numpy()
    compare(f"VAE decoder (PyTorch {vae_device})", decoded, expected, placement, 8e-4, 6e-4)
    if vae_device == "cuda:0":
        reference.to("cpu")
        torch.cuda.empty_cache()
    if not args.denoiser:
        return
    rng = np.random.default_rng(42)
    target = rng.standard_normal(latent.shape, dtype=np.float32)
    text = conditioning(args, work, image) if args.encoder else rng.standard_normal((9, 3584), dtype=np.float32)
    print(json.dumps({"image": [width, height], "target_tokens": target.size // 64,
                      "reference_tokens": latent.size // 64, "text_tokens": len(text)}), flush=True)
    mask = np.ones(len(text), dtype=np.int64)
    profile = "edit_2511_cuda_qk" if args.cuda_qk else "edit_2511"
    weights = work / "denoiser.safetensors"
    schema = schema_for(args.probe, profile)
    if shutil.disk_usage(work).free < 24 * 2**30:
        raise RuntimeError("At least 24 GiB of free temporary disk space is needed for the prepared denoiser")
    count = prepare(args.denoiser, weights, schema, "edit_2511")
    print(f"Qwen Image Edit 2511: {len(schema)} tensors and {count} scalar-scaled FP8 matrices", flush=True)
    times = np.array([0.78, 0.35, 0], dtype="<f4")
    baseline = None
    block_reference = None
    if args.block_diagnostics or args.qk_diagnostics:
        from block_diagnostics import BlockReference

        block_reference = BlockReference(projections=args.qk_diagnostics)
    if args.step_diagnostics:
        baseline = torch_full_denoiser(args.denoiser, schema, target, [latent], text, mask, times, devices[0],
                                       capture_trajectory=True, block_reference=block_reference)
        if devices[0] == "cuda:0":
            torch.cuda.empty_cache()
    if args.feedback_diagnostics:
        actual, denoiser_placement, samples = run_feedback(
            args, work, profile, weights, target, [latent], text, mask, times, baseline, sweep=not args.feedback_endpoints,
        )

        def reference_velocity(state):
            return torch_full_denoiser(args.denoiser, schema, state, [latent], text, mask, times[1:2], devices[0])

        reference_on_native_latent = report_feedback(samples, baseline, times, denoiser_placement, reference_velocity)
        if devices[0] == "cuda:0":
            torch.cuda.empty_cache()
    else:
        actual, denoiser_placement = run_denoiser(args, work, profile, weights, target, [latent], text, mask, times,
                                                  "edit-euler", reference_states=baseline.later_targets if baseline else None)
    step_checks = ()
    if baseline is not None:
        report_trajectory(actual, target, times, baseline, devices[0], denoiser_placement)
        if args.cuda_low_memory:
            step_checks = (
                ("First-step velocity", actual[1], baseline.velocities[0]),
                ("Second-step velocity on the PyTorch latent", actual[2], baseline.velocities[1]),
            )
            if args.feedback_diagnostics:
                _, fed_back, second_velocity = samples[-1]
                step_checks += (
                    ("First Euler latent", fed_back, baseline.later_targets[0]),
                    ("Second-step velocity on the same native latent", second_velocity, reference_on_native_latent),
                    ("Second-step velocity on the native trajectory", second_velocity, baseline.velocities[1]),
                )
        if args.block_diagnostics:
            from block_diagnostics import compare_blocks

            compare_blocks(args, run, work, profile, weights, target, [latent], text, mask, times[:1], block_reference,
                           baseline.velocities[0], actual[1])
        if args.qk_diagnostics:
            from block_diagnostics import errors
            from qk_diagnostics import compare_projections

            geometry = (target.size // 64, (target.size + latent.size) // 64, len(text), len(times) - 1)
            compare_projections(args, run, errors, work, weights, schema, block_reference.projections, geometry,
                                work / "edit-euler.input.f32")
        actual = actual[0]
    output, placement = vae_run(args, work, "vae_decode", decode_path, actual, h, w, "edit-rgb")
    png = work / "edit.png"
    pixels = np.rint(np.clip(output.transpose(1, 2, 0) * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(pixels).save(png)
    print(json.dumps({"size": [width, height], "steps": len(times) - 1, "qwen25_vl": bool(args.encoder),
                      "png_sha256": hashlib.sha256(png.read_bytes()).hexdigest()}), flush=True)
    failures = []
    references = {}
    for reference_device in devices:
        if baseline is not None and reference_device == devices[0]:
            reference_latent = baseline.final
        else:
            reference_latent = torch_full_denoiser(args.denoiser, schema, target, [latent], text, mask, times, reference_device)
        reference.to(reference_device)
        latent_input = image_tensor(reference_latent).to(reference_device)
        expected = reference.decode(normalizer.process_out(latent_input)).clamp(-1, 1)[0, :, 0].cpu().numpy()
        del latent_input
        if reference_device == "cuda:0":
            reference.to("cpu")
            torch.cuda.empty_cache()
        references[reference_device] = (reference_latent, expected)
        checks = (("Edited image RGB after independent denoising", output, expected, placement, 5e-3),
                  ("Edit 2511 Euler with reference image", actual, reference_latent, denoiser_placement, 4e-3))
        if reference_device == devices[0]:
            checks += tuple((label, value, ref, denoiser_placement, 4e-3) for label, value, ref in step_checks)
        for label, values, reference_values, device, atol in checks:
            try:
                use_elementwise = denoiser_placement.get("math") != "reference" or values is not output
                compare(f"{label} (PyTorch {reference_device})", values, reference_values, device, atol, 3e-3,
                        rmse_limit=1e-3, elementwise=use_elementwise)
            except AssertionError as error:
                failures.append(error)
    if "cpu" in references and "cuda:0" in references:
        cpu_latent, cpu_rgb = references["cpu"]
        cuda_latent, cuda_rgb = references["cuda:0"]
        report_delta("PyTorch CPU versus CUDA latents", cpu_latent, cuda_latent)
        report_delta("PyTorch CPU versus CUDA RGB", cpu_rgb, cuda_rgb)
    if failures:
        raise ExceptionGroup("Edit 2511 parity against PyTorch", failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("comfy", "probe", "vae", "denoiser", "encoder", "encoder-probe", "tokenizer"):
        parser.add_argument("--" + key, type=Path, required=key in ("probe", "comfy"))
    parser.add_argument("--backend", choices=PLATFORMS, default="cuda")
    parser.add_argument("--width", type=int, default=64, help="reference image width")
    parser.add_argument("--height", type=int, default=48, help="reference image height")
    parser.add_argument("--vision-size", type=int, help="square conditioning image size for Qwen2.5-VL")
    parser.add_argument("--expected-device")
    parser.add_argument("--expected-pjrt", help="exact PJRT platform to require, ignoring case")
    parser.add_argument("--cuda-low-memory", action="store_true",
                        help="run the Linux denoiser block by block with the CUDA BFC allocator")
    parser.add_argument("--reference-device", choices=("cpu", "cuda:0"), action="append",
                        help="device for the independent FP32 PyTorch transformer, repeatable (default: cpu)")
    parser.add_argument("--step-diagnostics", action="store_true",
                        help="evaluate both denoisers on the first PyTorch reference's exact trajectory")
    parser.add_argument("--block-diagnostics", action="store_true",
                        help="compare every block of the first denoising step, including with shared PyTorch hidden states")
    parser.add_argument("--qk-diagnostics", action="store_true",
                        help="verify the first block's projection, RMSNorm and RoPE against shared PyTorch CUDA inputs")
    parser.add_argument("--cuda-qk", action="store_true",
                        help="use the ordered first-block Q/K contractions for the CUDA denoiser")
    parser.add_argument("--feedback-diagnostics", action="store_true",
                        help="compare second-step velocities on both actual latents and on three interpolated latents")
    parser.add_argument("--feedback-endpoints", action="store_true",
                        help="compare second-step velocities only on the actual PyTorch and ZML latents")
    args = parser.parse_args()
    if any(size <= 0 or size > 1024 or size % 16 for size in (args.width, args.height)):
        parser.error("Image width and height must be multiples of 16 in 16..1024")
    if args.denoiser and 2 * (args.width // 16) * (args.height // 16) > 4096 and (
        not args.cuda_low_memory or args.cuda_qk or not os.getenv("ZML_QWEN_CUDA_MATH")
    ):
        parser.error("More than 4096 image tokens require the standard CUDA low-memory mode and ZML_QWEN_CUDA_MATH")
    if args.vision_size is not None and (not args.encoder or args.vision_size <= 0 or args.vision_size % 28
                                         or (args.vision_size // 14) ** 2 > 2048):
        parser.error("--vision-size needs an encoder and a multiple of 28 within the vision patch limit")
    args.feedback_diagnostics |= args.feedback_endpoints
    if args.denoiser and args.vae is None:
        parser.error("--denoiser requires --vae for the image reference")
    if args.cuda_low_memory and (sys.platform != "linux" or args.backend != "cuda" or not args.denoiser):
        parser.error("--cuda-low-memory requires Linux, --backend cuda and --denoiser")
    if args.cuda_low_memory and args.block_diagnostics:
        parser.error("--block-diagnostics requires the whole denoiser graph")
    if args.cuda_qk and (args.backend != "cuda" or not args.denoiser):
        parser.error("--cuda-qk requires --backend cuda and --denoiser")
    if (args.block_diagnostics or args.qk_diagnostics or args.step_diagnostics or args.feedback_diagnostics) and not args.denoiser:
        parser.error("Denoiser diagnostics require --denoiser")
    if args.qk_diagnostics and (args.backend != "cuda" or not args.reference_device or args.reference_device[0] != "cuda:0"):
        parser.error("--qk-diagnostics requires --backend cuda --reference-device cuda:0")
    args.step_diagnostics |= args.block_diagnostics or args.qk_diagnostics or args.feedback_diagnostics
    if any((args.encoder, args.encoder_probe, args.tokenizer)) and not all((args.encoder, args.encoder_probe, args.tokenizer, args.denoiser)):
        parser.error("Provide --encoder, --encoder-probe, --tokenizer and --denoiser together")
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve(strict=True))
    with scratch_directory() as work:
        os.environ.update({key: str(work) for key in ("TEMP", "TMP", "TMPDIR", "TORCHINDUCTOR_CACHE_DIR",
                                                      "TRITON_CACHE_DIR", "HF_HOME", "XDG_CACHE_HOME")})
        os.environ.update(CUDA_CACHE_DISABLE="1", HF_HUB_OFFLINE="1")
        sys.path.insert(0, str(args.comfy))
        sys.argv = [sys.argv[0], "--cpu"]
        import comfy.options
        import torch

        comfy.options.enable_args_parsing()
        torch.set_num_threads(8 if args.vae else 4)
        torch.set_float32_matmul_precision("highest")
        with torch.inference_mode():
            if args.vae:
                full(args, work)
            else:
                fixture(args, work)


if __name__ == "__main__":
    main()
