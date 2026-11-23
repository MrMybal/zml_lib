"""Compare T5 and UMT5 encoder outputs with Transformers using the same weights."""

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import T5Config, T5EncoderModel, UMT5Config, UMT5EncoderModel

from prepare import prepare, schema_for

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build import ROOT, run_environment, scratch_directory
from probe_placement import PLATFORMS, verify_placement


MODEL_TYPES = {
    "t5": (T5Config, T5EncoderModel, 32_128),
    "umt5": (UMT5Config, UMT5EncoderModel, 256_384),
}
FULL_PROFILES = tuple(f"{family}_xxl" for family in MODEL_TYPES)


def fixture_checkpoint(work, config_type, model_type):
    torch.manual_seed(197)
    config = config_type(vocab_size=80, d_model=32, d_kv=8, d_ff=64, num_layers=2, num_heads=4,
                         feed_forward_proj="gated-gelu", dropout_rate=0.0,
                         relative_attention_num_buckets=32, relative_attention_max_distance=128)
    model = model_type(config).eval()
    source = work / "reference.safetensors"
    save_file({name: value.detach().contiguous() for name, value in model.named_parameters()}, source)
    return source, config


def cases_from_transformers(profile, weights, schema, lengths, fixture_config):
    family = profile.split("_", 1)[0]
    config_type, model_type, vocab = MODEL_TYPES[family]
    full = fixture_config is None
    config = fixture_config if not full else config_type(
        vocab_size=vocab, d_model=4096, d_kv=64, d_ff=10240, num_layers=24, num_heads=64,
        feed_forward_proj="gated-gelu", dropout_rate=0.0, layer_norm_epsilon=1e-6,
        relative_attention_num_buckets=32, relative_attention_max_distance=128,
    )
    with torch.device("meta"):
        model = model_type(config).eval()
    state = dict(model.named_parameters())
    mismatched = [name for name in state.keys() & schema.keys() if tuple(state[name].shape) != schema[name]]
    if state.keys() != schema.keys() or mismatched:
        raise AssertionError(f"Unexpected schema: missing={state.keys() - schema.keys()}, "
                             f"extra={schema.keys() - state.keys()}, shapes={mismatched}")
    with safe_open(weights, framework="pt", device="cpu") as stored:
        params = {name: stored.get_tensor(name).float() for name in schema}
    params["encoder.embed_tokens.weight"] = params["shared.weight"]
    model.load_state_dict(params, assign=True)
    cases = []
    for length in lengths:
        ids = (torch.arange(length) * (104_729 if full else 7) + 3) % config.vocab_size
        mask = (torch.arange(1, length + 1) % 13 != 0).int()
        with torch.inference_mode():
            expected = model(input_ids=ids[None], attention_mask=mask[None]).last_hidden_state.numpy()
        if not np.isfinite(expected).all():
            raise AssertionError(f"Nonfinite Transformers output for {profile}, tokens={length}")
        cases.append((torch.stack((ids, mask)).numpy().astype("<f4"), expected))
    return cases


def replay(args, profile, work):
    env = run_environment(work, args.tools)
    schema = schema_for(args.probe, profile)
    fixture_config = None
    source = args.source
    if source is None:
        config_type, model_type, _ = MODEL_TYPES[profile.split("_", 1)[0]]
        source, fixture_config = fixture_checkpoint(work, config_type, model_type)
    weights = work / "input.safetensors"
    print(f"{profile} {args.backend}: preparing {args.storage} weights", flush=True)
    prepare(source, weights, schema, args.storage)
    print(f"{profile} {args.backend}: evaluating the FP32 Transformers reference", flush=True)
    cases = cases_from_transformers(profile, weights, schema, args.length or (9, 149), fixture_config)
    for inputs, expected in cases:
        length = inputs.shape[1]
        token_file = work / f"tokens-{length}.f32le"
        output = work / f"encoder-{length}.f32le"
        inputs.tofile(token_file)
        command = [args.probe, profile, f"--storage={args.storage}", args.backend,
                   weights, str(length), token_file, output]
        result = subprocess.run(command, env=env, capture_output=True, text=True, errors="replace")
        print(result.stderr, end="", file=sys.stderr)
        result.check_returncode()
        verify_placement(result.stderr, args.backend, 2, args.expected_device)
        observed = np.fromfile(output, dtype="<f4").reshape(expected.shape)
        delta = np.abs(observed - expected)
        print(f"{profile} {args.backend} storage={args.storage} tokens={length} "
              f"max_abs={delta.max():.9g} rmse={np.sqrt(np.mean(delta ** 2)):.9g}", flush=True)
        atol, rtol = (4e-5, 4e-5) if fixture_config is not None else (4e-4, 3e-4)
        np.testing.assert_allclose(observed, expected, atol=atol, rtol=rtol, equal_nan=False)
        token_file.unlink()
        output.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--backend", choices=PLATFORMS, default="cpu")
    parser.add_argument("--expected-device")
    parser.add_argument("--profile", choices=FULL_PROFILES)
    parser.add_argument("--source", type=Path, help="local SafeTensors checkpoint or sharded directory")
    parser.add_argument("--storage", choices=("f32", "f16"), default="f32")
    parser.add_argument("--length", type=int, action="append", help="test lengths, defaults to 9 and 149")
    args = parser.parse_args()
    args.probe = args.probe.resolve(strict=True)
    if (args.source is None) != (args.profile is None):
        parser.error("Specify --source with its --profile for an XXL checkpoint")
    if args.source is not None:
        args.source = args.source.resolve(strict=True)
    if args.length and any(not 1 <= length <= 512 for length in args.length):
        parser.error("Each test length must be between 1 and 512 tokens")
    args.tools = ROOT.parent / "zml_win" / "bazel-win-tools"
    torch.set_num_threads(8 if args.source else 4)
    profiles = (args.profile,) if args.profile else ("t5_fixture", "umt5_fixture")
    for profile in profiles:
        with scratch_directory() as work:
            replay(args, profile, work)


if __name__ == "__main__":
    main()
