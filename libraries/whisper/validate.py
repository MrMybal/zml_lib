"""Compare the native Whisper encoder and next-token logits against Transformers."""

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import WhisperConfig, WhisperForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from prepare import prepare
from transcribe import open_worker

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build import ROOT, run_environment, scratch_directory
from probe_placement import PLATFORMS, verify_placement


def native(args, profile, stage, data, weights, work, env, shape, *, count=None):
    source = work / f"{stage}-{count}.f32le"
    destination = work / f"{stage}-{count}-out.f32le"
    data.astype("<f4", copy=False).tofile(source)
    command = [args.probe, profile, stage, args.backend, weights / f"{stage}.safetensors", source, destination]
    if count is not None:
        command.append(str(count))
    result = subprocess.run(command, env=env, capture_output=True, text=True, errors="replace")
    print(result.stderr, end="", file=sys.stderr)
    result.check_returncode()
    verify_placement(result.stderr, args.backend, 1, args.expected_device)
    return np.fromfile(destination, dtype="<f4").reshape(shape)


@torch.inference_mode()
def validate(args, work):
    env = run_environment(work, ROOT.parent / "zml_win" / "bazel-win-tools")
    if args.model:
        model = WhisperForConditionalGeneration.from_pretrained(args.model, local_files_only=True,
                                                                dtype=torch.float32, attn_implementation="eager").eval()
        source = args.model / "model.safetensors"
    else:
        torch.manual_seed(442)
        config = WhisperConfig(vocab_size=96, num_mel_bins=8, d_model=32, encoder_layers=2, decoder_layers=2,
                               encoder_attention_heads=4, decoder_attention_heads=4, encoder_ffn_dim=64,
                               decoder_ffn_dim=64, max_source_positions=16, max_target_positions=24,
                               bos_token_id=1, eos_token_id=2, pad_token_id=2, decoder_start_token_id=1)
        model = WhisperForConditionalGeneration(config).eval()
        source = work / "reference.safetensors"
        save_file({name: value.detach().contiguous() for name, value in model.named_parameters()}, source)
    config = model.config
    profile = args.profile
    weights = work / "weights"
    prepare(args.probe, profile, source, weights)
    rng = np.random.default_rng(967)
    mels = rng.normal(0, 0.6, (1, config.num_mel_bins, config.max_source_positions * 2)).astype(np.float32)
    reference = model.model.encoder(torch.from_numpy(mels)).last_hidden_state.numpy()
    encoded = native(args, profile, "encoder", mels, weights, work, env, reference.shape)
    tolerance = 8e-5 if not args.model else 1e-3
    print(f"Whisper {profile} encoder max_abs={np.abs(encoded - reference).max():.9g}", flush=True)
    np.testing.assert_allclose(encoded, reference, atol=tolerance, rtol=tolerance, equal_nan=False)

    prompt = [config.decoder_start_token_id, 5, 7]
    for _ in range(2):
        expected = model(encoder_outputs=BaseModelOutput(last_hidden_state=torch.from_numpy(reference)),
                         decoder_input_ids=torch.tensor([prompt]),
                         use_cache=False).logits[:, -1, :].numpy()
        data = np.concatenate((encoded.reshape(-1), np.asarray(prompt, dtype=np.float32)))
        scores = native(args, profile, "decoder", data, weights, work, env, expected.shape, count=len(prompt))
        token = int(scores.argmax())
        print(f"Whisper {profile} decoder prefix={len(prompt)} max_abs={np.abs(scores - expected).max():.9g} "
              f"next_token={token}", flush=True)
        np.testing.assert_allclose(scores, expected, atol=tolerance, rtol=tolerance, equal_nan=False)
        if token != int(expected.argmax()):
            raise AssertionError("Whisper decoded a different token")
        prompt.append(token)

    if args.profile == "fixture":
        with open_worker(args.probe, profile, weights, args.backend, 12, work, env, args.expected_device) as worker:
            for features in (mels, mels[:, :, ::-1].copy()):
                worker.encode(features)
                for length in (3, len(prompt)):
                    prefix = prompt[:length]
                    expected = model(input_features=torch.from_numpy(features), decoder_input_ids=torch.tensor([prefix]),
                                     use_cache=False).logits[0, -1].numpy()
                    actual = worker.scores(prefix)
                    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance, equal_nan=False)
                    print(f"Whisper persistent prefix={length} max_abs={np.abs(actual - expected).max():.9g}", flush=True)
            if worker.windows != 2 or worker.steps != 4:
                raise AssertionError("Expected two windows and four steps in one native process")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--profile", choices=("fixture", "tiny", "large_v3_turbo"), default="fixture")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--backend", choices=PLATFORMS, default="cpu")
    parser.add_argument("--expected-device")
    args = parser.parse_args()
    if bool(args.model) != (args.profile != "fixture"):
        parser.error("Provide --model when validating an official profile")
    args.probe = args.probe.resolve(strict=True)
    torch.set_num_threads(4)
    with scratch_directory() as work:
        validate(args, work)


if __name__ == "__main__":
    main()
