"""Check and convert a local Qwen3-TTS 12 Hz checkpoint for a ZML stage."""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


PROFILES = {"qwen3_tts_0b6": ("0b6", 1024), "qwen3_tts_1b7": ("1b7", 2048)}
STAGES = ("text", "codec", "talker", "predictor")
LINUX_CUDA_FLAGS = "--xla_gpu_enable_triton_gemm=false --xla_gpu_enable_cublaslt=false"


def native_env(backend):
    env = os.environ.copy()
    if sys.platform == "linux" and backend == "cuda":
        env.setdefault("XLA_FLAGS", LINUX_CUDA_FLAGS)
        env.setdefault("NVIDIA_TF32_OVERRIDE", "0")
    return env


def verify_config(directory, profile):
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    size, width = PROFILES[profile]
    expected = {"model_type": "qwen3_tts", "tts_model_size": size, "tokenizer_type": "qwen3_tts_tokenizer_12hz"}
    shared = {"num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
              "num_code_groups": 16, "rms_norm_eps": 1e-6, "rope_theta": 1_000_000,
              "attention_bias": False, "sliding_window": None, "hidden_act": "silu"}
    talker_fields = dict(shared, hidden_size=width, intermediate_size=3 * width, num_hidden_layers=28,
                         text_hidden_size=2048, text_vocab_size=151936, vocab_size=3072)
    predictor_fields = dict(shared, hidden_size=1024, intermediate_size=3072, num_hidden_layers=5,
                            vocab_size=2048, rope_scaling=None)
    talker = config["talker_config"]
    predictor = talker["code_predictor_config"]
    for section, fields in ((config, expected), (talker, talker_fields), (predictor, predictor_fields)):
        for name, value in fields.items():
            if section.get(name) != value:
                raise ValueError(f"{name}: expected {value!r}, got {section.get(name)!r}")
        if section.get("use_sliding_window") or any(
            kind != "full_attention" for kind in (section.get("layer_types") or ())
        ):
            raise ValueError("Expected full causal attention")
    rope = talker["rope_scaling"]
    if rope.get("rope_type", rope.get("type")) != "default" or rope.get("mrope_section") != [24, 20, 20]:
        raise ValueError("Expected the Qwen3-TTS Talker rotary configuration")
    return config


def source_name(stage, name):
    if stage == "speaker":
        return name
    if stage == "text" and name == "text_embedding.weight":
        return "talker.model." + name
    if stage == "codec":
        prefix = "talker.model." if name == "codec_embedding.weight" else "talker.code_predictor.model."
    else:
        prefix = "talker.code_predictor." if stage == "predictor" else "talker."
    return prefix + name


def verify_speaker_config(directory, profile):
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if config.get("tts_model_type") != "base":
        raise ValueError("The speaker encoder is available on Qwen3-TTS Base")
    speaker = config["speaker_encoder_config"]
    defaults = {
        "sample_rate": 24000, "mel_dim": 128, "enc_dim": 1024,
        "enc_channels": [512, 512, 512, 512, 1536], "enc_kernel_sizes": [5, 3, 3, 3, 1],
        "enc_dilations": [1, 2, 3, 4, 1], "enc_res2net_scale": 8,
        "enc_attention_channels": 128, "enc_se_channels": 128,
    }
    expected = dict(defaults, enc_dim=PROFILES[profile][1])
    for name, value in expected.items():
        actual = speaker.get(name, defaults[name])
        if actual != value:
            raise ValueError(f"Speaker {name}: expected {value!r}, got {actual!r}")


def prepare(directory, profile, stage, schema, output=None):
    import torch
    from safetensors import safe_open

    verify_config(directory, profile)
    if stage == "speaker":
        verify_speaker_config(directory, profile)
    storage, dtype = ("f32", torch.float32) if stage == "speaker" else ("f16", torch.float16)
    with ExitStack() as stack:
        shards = {}
        for path in sorted(directory.glob("model*.safetensors")):
            checkpoint = stack.enter_context(safe_open(path, framework="pt", device="cpu"))
            for name in checkpoint.keys():
                if name in shards:
                    raise ValueError(f"Duplicate checkpoint tensor {name}")
                shards[name] = checkpoint
        selected = {}
        for name, shape in schema.items():
            source = source_name(stage, name)
            if source not in shards:
                raise ValueError(f"Missing checkpoint tensor {source}")
            item = shards[source].get_slice(source)
            if tuple(item.get_shape()) != shape or item.get_dtype() not in {"BF16", "F16", "F32"}:
                raise ValueError(f"{source}: expected {shape} as BF16, F16 or F32, got {item.get_shape()} {item.get_dtype()}")
            selected[name] = source
        if output is None:
            return

        def tensors():
            for name in selected.values():
                value = shards[name].get_tensor(name).to(dtype)
                if not torch.isfinite(value).all():
                    raise ValueError(f"Nonfinite {storage.upper()} value in {name}")
                yield value

        publish_checkpoint(output, storage, schema, tensors())


def publish_checkpoint(output, storage, schema, values):
    from libraries.flux1.prepare import write_tensors

    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    with TemporaryDirectory(prefix="qwen-tts-prepare-", dir=output.parent) as temp:
        staged = Path(temp) / "model.safetensors"
        fields = {name: (shape, storage) for name, shape in schema.items()}
        write_tensors(staged, storage, fields, values)
        output.hardlink_to(staged)


def prepare_codec_weights(directory, schema, output):
    import torch
    from safetensors import safe_open
    from libraries.flux1.prepare import selected_tensors

    with safe_open(directory / "model.safetensors", framework="pt", device="cpu") as checkpoint:
        selected = selected_tensors(checkpoint, schema, "")
        for name in selected.values():
            if checkpoint.get_slice(name).get_dtype() not in {"F32", "F16", "BF16"}:
                raise ValueError(f"{name}: expected F32, F16 or BF16")
        if output is None:
            return

        def tensors():
            for name in selected.values():
                value = checkpoint.get_tensor(name).to(torch.float32)
                if not torch.isfinite(value).all():
                    raise ValueError(f"Nonfinite FP32 value in {name}")
                yield value

        publish_checkpoint(output, "f32", schema, tensors())


def prepare_waveform(directory, schema, output=None):
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    fields = {
        "model_type": "qwen3_tts_tokenizer_12hz", "output_sample_rate": 24000, "decode_upsample_rate": 1920,
    }
    decoder = {
        "codebook_size": 2048, "codebook_dim": 512, "latent_dim": 1024, "decoder_dim": 1536,
        "hidden_size": 512, "intermediate_size": 1024, "head_dim": 64, "num_attention_heads": 16,
        "num_key_value_heads": 16, "num_hidden_layers": 8, "num_quantizers": 16, "num_semantic_quantizers": 1,
        "attention_bias": False, "rms_norm_eps": 1e-5, "rope_theta": 10000, "hidden_act": "silu",
        "sliding_window": 72, "upsampling_ratios": [2, 2], "upsample_rates": [8, 5, 4, 3],
    }
    for section, expected in ((config, fields), (config["decoder_config"], decoder)):
        for name, value in expected.items():
            if section.get(name) != value:
                raise ValueError(f"{name}: expected {value!r}, got {section.get(name)!r}")
    if config["decoder_config"].get("rope_scaling") is not None or any(
        kind != "sliding_attention" for kind in (config["decoder_config"].get("layer_types") or ())
    ):
        raise ValueError("Expected default RoPE and sliding attention in the 12 Hz decoder")

    prepare_codec_weights(directory, schema, output)


def prepare_audio_encoder(directory, schema, output=None):
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    expected = {
        "model_type": "qwen3_tts_tokenizer_12hz", "input_sample_rate": 24000,
        "encode_downsample_rate": 1920, "encoder_valid_num_quantizers": 16,
    }
    encoder = {
        "audio_channels": 1, "sampling_rate": 24000, "_frame_rate": 12.5, "num_filters": 64,
        "num_residual_layers": 1, "compress": 2, "kernel_size": 7, "residual_kernel_size": 3,
        "last_kernel_size": 3, "use_conv_shortcut": False, "pad_mode": "constant",
        "use_causal_conv": True, "use_streaming": False, "upsampling_ratios": [8, 6, 5, 4],
        "hidden_size": 512, "head_dim": 64, "num_attention_heads": 8, "num_key_value_heads": 8,
        "num_hidden_layers": 8, "intermediate_size": 2048, "norm_eps": 1e-5,
        "hidden_act": "gelu", "attention_bias": False, "rope_theta": 10000, "normalize": False,
        "codebook_size": 2048, "codebook_dim": 256, "vector_quantization_hidden_dimension": 256,
        "num_semantic_quantizers": 1, "num_quantizers": 32,
    }
    for section, fields in ((config, expected), (config["encoder_config"], encoder)):
        for name, value in fields.items():
            if section.get(name) != value:
                raise ValueError(f"Audio encoder {name}: expected {value!r}, got {section.get(name)!r}")
    if config["encoder_config"].get("rope_scaling") is not None:
        raise ValueError("Expected default Mimi RoPE")
    prepare_codec_weights(directory, schema, output)


def main():
    from libraries.flux1.prepare import schema_for

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--stage", choices=(*STAGES, "speaker", "audio_encoder", "waveform"), required=True)
    parser.add_argument("--output", type=Path, help="omit to verify the checkpoint without writing")
    args = parser.parse_args()
    probe = args.probe.resolve(strict=True)
    model = args.model.resolve(strict=True)
    if args.stage in ("waveform", "audio_encoder"):
        if args.profile is not None:
            parser.error(f"--stage {args.stage} uses the shared speech tokenizer without --profile")
        schema = schema_for(probe, args.stage)
        convert = prepare_waveform if args.stage == "waveform" else prepare_audio_encoder
        convert(model, schema, args.output)
    else:
        if args.profile is None:
            parser.error("--profile is required for Talker stages")
        schema = schema_for(probe, args.profile, args.stage)
        prepare(model, args.profile, args.stage, schema, args.output)
    print(json.dumps({"profile": args.profile, "stage": args.stage, "tensors": len(schema)}))


if __name__ == "__main__":
    main()
