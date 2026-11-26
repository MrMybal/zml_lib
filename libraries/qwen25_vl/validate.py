"""Compare the Qwen2.5-VL encoders with local Transformers and verify device placement."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from inputs import image_positions, text_records, vision_records

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.build import scratch_directory
from scripts.probe_placement import PLATFORMS, verify_placement


def native(args, work, profile, weights, inputs, output_width):
    count = len(inputs[0])
    command = [args.probe, profile, args.backend, weights, str(count)]
    outputs = []
    for index, records in enumerate(inputs):
        if len(records) != count:
            raise ValueError("A repeated native call must use the same sequence length")
        source = work / f"{profile}-{index}.input.f32"
        destination = work / f"{profile}-{index}.output.f32"
        records.tofile(source)
        command.extend((source, destination))
        outputs.append(destination)
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(result.stderr, end="", file=sys.stderr)
    result.check_returncode()
    placement = verify_placement(result.stderr, args.backend, len(inputs), args.expected_device)
    divisor = 4 if profile == "vision" else 1
    observed = [np.fromfile(path, dtype="<f4").reshape(1, count // divisor, output_width) for path in outputs]
    return observed, placement


def compare(name, observed, expected, placement, tolerance, *, mask=None, atol=None, rmse_limit=None):
    shape = list(observed.shape)
    if mask is not None:
        padding_delta = np.abs(observed[:, ~mask] - expected[:, ~mask])
        if padding_delta.size:
            print(f"{name}: {padding_delta.shape[1]} ignored padding tokens, max_abs={padding_delta.max():.9g}", flush=True)
        observed, expected = observed[:, mask], expected[:, mask]
    delta = np.abs(observed - expected)
    rmse = float(np.sqrt(np.mean(delta**2)))
    summary = {"case": name, "shape": shape, "checked_tokens": observed.shape[1], "max_abs": float(delta.max()),
               "rmse": rmse, "placement": placement}
    print(json.dumps(summary, allow_nan=False), flush=True)
    np.testing.assert_allclose(observed, expected, atol=tolerance if atol is None else atol,
                               rtol=tolerance, equal_nan=False)
    if rmse_limit is not None and rmse > rmse_limit:
        raise AssertionError(f"RMSE {rmse:.9g} exceeds {rmse_limit:.9g}")


def load_reference(model, schema, path, prefix):
    from safetensors import safe_open

    layout = {prefix + key: tuple(tensor.shape) for key, tensor in model.state_dict().items()}
    if schema != layout:
        raise AssertionError(f"Reference schema differs for {prefix}; extra={schema.keys() - layout.keys()}, "
                             f"missing={layout.keys() - schema.keys()}")
    with safe_open(path, framework="pt", device="cpu") as store:
        weights = {key.removeprefix(prefix): store.get_tensor(key).float() for key in schema}
    model.load_state_dict(weights, strict=True, assign=True)


def text_config(fixture):
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig

    return Qwen2_5_VLTextConfig(
        vocab_size=64 if fixture else 152064, hidden_size=32 if fixture else 3584,
        intermediate_size=48 if fixture else 18944, num_attention_heads=4 if fixture else 28,
        num_key_value_heads=2 if fixture else 4, num_hidden_layers=2 if fixture else 28,
        hidden_act="silu", rms_norm_eps=1e-6, rope_theta=1_000_000, use_sliding_window=False,
        max_position_embeddings=128000, use_cache=False, attn_implementation="eager",
        rope_scaling={"type": "mrope", "mrope_section": [2, 1, 1] if fixture else [16, 24, 24]},
    )


def text_reference(model, records):
    import torch

    ids = torch.from_numpy(records[:, 0].astype(np.int64))
    tokens = model.embed_tokens(ids.clamp(min=0))
    tokens[ids == -1] = torch.from_numpy(records[ids.numpy() == -1, 5:].copy())
    positions = torch.tensor(records[:, 1:4].T.astype(np.int64))[:, None, :]
    mask = torch.tensor(records[None, :, 4].astype(np.int64))
    return model(inputs_embeds=tokens[None], position_ids=positions, attention_mask=mask).last_hidden_state.numpy()


def fixture(args, work):
    import torch
    from safetensors.torch import save_file
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel

    from prepare import read_schema

    torch.manual_seed(275)
    model = Qwen2_5_VLTextModel(text_config(True)).eval()
    schema = read_schema(args.probe, "fixture")
    weights = work / "fixture.safetensors"
    save_file({"model." + key: value.half().contiguous() for key, value in model.state_dict().items()}, weights)
    load_reference(model, schema, weights, "model.")
    ids = [8, 19, 22, 41, 63, 37, 2, 5, 17]
    text = text_records(ids, 32)
    pos = np.repeat(np.arange(9)[:, None], 3, axis=1)
    pos[2:5] = [[2, 4, 2], [2, 56, 13], [2, 45, 76]]
    random = np.random.default_rng(25)
    mixed = text_records(ids, 32, positions=pos, attention_mask=[0, 1, 1, 1, 1, 1, 1, 1, 0],
                         embeddings={i: random.normal(size=32) for i in (2, 3, 4)})
    observed, placement = native(args, work, "fixture", weights, [text, mixed], 32)
    for label, values, result in zip(("fixture text", "fixture MRoPE and padding"), (text, mixed), observed, strict=True):
        compare(label, result, text_reference(model, values), placement, 8e-5)

    malformed = mixed.copy()
    malformed[2, 1] = -1
    input_path, output_path = work / "invalid.f32", work / "invalid.out"
    malformed.tofile(input_path)
    failure = subprocess.run([args.probe, "fixture", args.backend, weights, str(len(malformed)),
                              input_path, output_path], capture_output=True, text=True, errors="replace")
    if not failure.returncode or "InvalidPosition" not in failure.stderr or output_path.exists():
        raise AssertionError("Malformed MRoPE positions should fail before inference")


def verify_conversion(source, converted, layer):
    import torch
    from safetensors import safe_open

    with safe_open(source, framework="pt") as raw, safe_open(converted, framework="pt") as output:
        values = raw.get_tensor(layer + ".weight").float()
        expected = (values * raw.get_tensor(layer + ".scale_weight").item()).half()
        if not torch.equal(expected, output.get_tensor(layer + ".weight")):
            raise AssertionError(f"Wrong scaled FP8 conversion for {layer}")


def vision_test(args, work, weights, schema):
    import torch
    from PIL import Image, ImageDraw, ImageOps
    from transformers import Qwen2VLImageProcessor
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VisionRotaryEmbedding,
        Qwen2_5_VisionTransformerPretrainedModel,
    )

    config = Qwen2_5_VLVisionConfig(hidden_size=1280, tokens_per_second=2, attn_implementation="eager")
    with torch.device("meta"):
        model = Qwen2_5_VisionTransformerPretrainedModel(config).eval()
    model.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(40)
    load_reference(model, schema, weights, "visual.")
    image = Image.new("RGB", (196, 140), (15, 48, 91))
    draw = ImageDraw.Draw(image)
    draw.rectangle((18, 21, 114, 115), fill=(241, 129, 56))
    draw.ellipse((126, 25, 174, 85), fill=(49, 215, 144))
    processor = Qwen2VLImageProcessor()
    inputs, reference = [], []
    for frame in (image, ImageOps.invert(image)):
        values = processor(images=frame, do_resize=False, return_tensors="pt")
        grid = values.image_grid_thw
        inputs.append(vision_records(values.pixel_values.numpy(), grid.numpy()))
        reference.append(model(values.pixel_values, grid_thw=grid).numpy()[None])
    actual, placement = native(args, work, "vision", weights, inputs, 3584)
    for index, (found, expected) in enumerate(zip(actual, reference, strict=True), start=1):
        compare(f"vision image {index}", found, expected, placement, 5e-4)
    return grid.numpy()[0, 1:] // 2, actual[0][0], reference[0][0]


def full(args, work):
    import torch
    from transformers import Qwen2Tokenizer
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLModel,
        Qwen2_5_VLRotaryEmbedding,
        Qwen2_5_VLTextModel,
    )

    from prepare import prepare, read_schema

    vision_path = work / "vision.safetensors"
    vision_schema = read_schema(args.probe, "vision")
    count = prepare(args.source, vision_path, vision_schema, "vision")
    print(f"Vision: {len(vision_schema)} tensors, {count} scaled E4M3FN matrices", flush=True)
    verify_conversion(args.source, vision_path, "visual.blocks.0.attn.qkv")
    shape, native_image, reference_image = vision_test(args, work, vision_path, vision_schema)
    vision_path.unlink()

    profile = "qwen25_vl_7b"
    text_path = work / "text.safetensors"
    schema = read_schema(args.probe, profile)
    count = prepare(args.source, text_path, schema, "text")
    print(f"Text: {len(schema)} tensors, {count} scaled E4M3FN matrices", flush=True)
    verify_conversion(args.source, text_path, "model.layers.0.self_attn.q_proj")
    config = text_config(False)
    with torch.device("meta"):
        model = Qwen2_5_VLTextModel(config).eval()
    model.rotary_emb = Qwen2_5_VLRotaryEmbedding(config)
    load_reference(model, schema, text_path, "model.")
    tokenizer = Qwen2Tokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    prompt = ("<|im_start|>system\nDescribe the image.<|im_end|>\n<|im_start|>user\n"
              "<|vision_start|><|image_pad|><|vision_end|>Replace the orange shape with a purple square."
              "<|im_end|>\n<|im_start|>assistant\n")
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    start = ids.index(151655)
    spatial_height, spatial_width = map(int, shape)
    image_token_count = spatial_height * spatial_width
    expanded = ids[:start] + [151655] * image_token_count + ids[start + 1:]
    pos = image_positions(start, spatial_height, spatial_width, len(ids) - start - 1)
    rope_model = SimpleNamespace(config=Qwen2_5_VLConfig())
    reference_pos, _ = Qwen2_5_VLModel.get_rope_index(rope_model, torch.tensor([expanded]),
                                                     image_grid_thw=torch.tensor([[1, spatial_height * 2, spatial_width * 2]]))
    np.testing.assert_array_equal(pos, reference_pos[:, 0, :].T.numpy())
    text_ids = tokenizer("A purple square on a navy blue background.", add_special_tokens=False).input_ids
    text_only = text_records(text_ids + [151643] * (len(expanded) - len(text_ids)),
                             attention_mask=[1] * len(text_ids) + [0] * (len(expanded) - len(text_ids)))
    with_native_image = text_records(expanded, positions=pos, embeddings=dict(enumerate(native_image, start)))
    with_reference_image = text_records(expanded, positions=pos, embeddings=dict(enumerate(reference_image, start)))
    actual, placement = native(args, work, profile, text_path, [text_only, with_native_image], 3584)
    cases = (
        ("Qwen2.5-VL text", actual[0], text_only, text_only[:, 4].astype(bool)),
        ("Qwen2.5-VL multimodal from the same embeddings", actual[1], with_native_image, None),
        ("Qwen2.5-VL image to conditioning", actual[1], with_reference_image, None),
    )
    failures = []
    for label, output, records, mask in cases:
        try:
            compare(label, output, text_reference(model, records), placement, 2e-3,
                    mask=mask, atol=4e-3, rmse_limit=2.5e-4)
        except AssertionError as error:
            failures.append(f"{label}: {error}")
    if failures:
        raise AssertionError("\n".join(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--backend", choices=PLATFORMS, default="cuda")
    parser.add_argument("--expected-device")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    args = parser.parse_args()
    if (args.source is None) != (args.tokenizer is None):
        parser.error("Supply both --source and --tokenizer for the full local checkpoint")
    args.probe = args.probe.resolve(strict=True)
    with scratch_directory() as work:
        os.environ.update({key: str(work) for key in ("TEMP", "TMP", "TMPDIR", "TORCHINDUCTOR_CACHE_DIR",
                                                      "TRITON_CACHE_DIR", "HF_HOME")})
        os.environ.update(CUDA_CACHE_DISABLE="1", HF_HUB_OFFLINE="1")
        import torch

        torch.set_num_threads(8 if args.source else 4)
        with torch.inference_mode():
            if args.source:
                full(args, work)
            else:
                fixture(args, work)


if __name__ == "__main__":
    main()
