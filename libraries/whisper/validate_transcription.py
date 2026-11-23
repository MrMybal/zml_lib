"""Compare the Whisper ZML CLI with Transformers on a short and a two-window audio file."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from prepare import prepare
from transcribe import CHUNK_SAMPLES, SAMPLE_RATE, TIMESTAMP_SAMPLES, decoder_prompt, read_audio, segment_tokens

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build import ROOT, run_environment, scratch_directory
from probe_placement import PLATFORMS


@torch.inference_mode()
def reference(model_dir, audio, language, limit):
    processor = WhisperProcessor.from_pretrained(model_dir, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_dir, local_files_only=True, dtype=torch.float32, attn_implementation="eager",
    ).eval()
    config = model.generation_config
    prefix = decoder_prompt(processor, config, language)
    features = processor.feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
    ids = model.generate(
        features, language=language, task="transcribe", return_timestamps=True,
        do_sample=False, temperature=0.0, condition_on_prev_tokens=False,
        max_new_tokens=limit, force_unique_generate_call=True,
    )[0].tolist()
    if ids[:len(prefix)] == prefix:
        ids = ids[len(prefix):]
    if not ids or ids[-1] != config.eos_token_id:
        raise AssertionError(f"Transformers did not finish the test transcription: {ids}")
    ids.pop()
    decoded = processor.tokenizer.decode(ids, skip_special_tokens=True, output_offsets=True)
    if not decoded["offsets"] or not decoded["text"].strip():
        raise AssertionError(f"Test audio must produce text with timestamps: {decoded}")
    spans = [{"start": item["timestamp"][0], "end": item["timestamp"][1],
              "text": item["text"].strip()} for item in decoded["offsets"] if item["text"].strip()]
    return ids, spans


def native(args, audio, weights, env):
    script = Path(__file__).with_name("transcribe.py")
    command = [sys.executable, "-B", script, "--audio", audio, "--model", args.model, "--weights", weights,
               "--profile", args.profile, "--probe", args.probe, "--backend", args.backend,
               "--language", args.language, "--max-new-tokens", str(args.limit)]
    if args.expected_device:
        command += ["--expected-device", args.expected_device]
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(result.stderr, end="", file=sys.stderr)
    result.check_returncode()
    return json.loads(result.stdout)


def compare(result, ids, expected, copies):
    if result["token_windows"] != [ids] * copies or not result["complete"]:
        raise AssertionError(f"Different token sequence or incomplete native generation: {result}")
    spans = [{"start": min(result["duration"], entry["start"] + chunk * 30),
              "end": min(result["duration"], entry["end"] + chunk * 30), "text": entry["text"]}
             for chunk in range(copies) for entry in expected]
    if len(result["segments"]) != len(spans) or result["text"] != " ".join(entry["text"] for entry in spans):
        raise AssertionError(f"Different segments from Transformers: {result}")
    for actual, target in zip(result["segments"], spans, strict=True):
        if actual["text"] != target["text"] or not 0 <= actual["start"] <= actual["end"] <= result["duration"]:
            raise AssertionError(f"Invalid native segment {actual}")
        np.testing.assert_allclose([actual["start"], actual["end"]], [target["start"], target["end"]],
                                   atol=0.0001, rtol=0)


def check_incomplete_segment():
    class Tokenizer:
        def decode(self, tokens, *, skip_special_tokens):
            return "".join({3: "hello", 4: "from", 5: "Whisper"}[token] for token in tokens)

    tokens = [100, 3, 120, 120, 4]
    closed, advance = segment_tokens(tokens, Tokenizer(), 100, CHUNK_SAMPLES, final_window=False)
    boundary = 20 * TIMESTAMP_SAMPLES
    assert closed == [(0, boundary, "hello")] and advance == boundary
    final, advance = segment_tokens(tokens, Tokenizer(), 100, CHUNK_SAMPLES, final_window=True)
    assert final == [(0, boundary, "hello"), (boundary, CHUNK_SAMPLES, "from")] and advance == CHUNK_SAMPLES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--profile", choices=("tiny", "large_v3_turbo"), default="large_v3_turbo")
    parser.add_argument("--language", default="en")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--backend", choices=PLATFORMS, default="cpu")
    parser.add_argument("--expected-device")
    args = parser.parse_args()
    args.model = args.model.resolve(strict=True)
    args.audio = args.audio.resolve(strict=True)
    torch.set_num_threads(6)
    check_incomplete_segment()
    audio = read_audio(args.audio)
    if len(audio) >= CHUNK_SAMPLES:
        parser.error("Use a spoken fixture shorter than 30 seconds")
    ids, expected = reference(args.model, audio, args.language, args.limit)
    print(json.dumps({"reference_ids": ids, "reference_segments": expected}), flush=True)
    if args.probe is None:
        return
    args.probe = args.probe.resolve(strict=True)
    tools = ROOT.parent / "zml_win" / "bazel-win-tools"
    with scratch_directory() as work:
        env = run_environment(work, tools)
        env["PYTHONUTF8"] = "1"
        weights = work / "weights"
        prepare(args.probe, args.profile, args.model / "model.safetensors", weights)
        short = native(args, args.audio, weights, env)
        compare(short, ids, expected, 1)
        print(json.dumps({"case": "short", "backend": args.backend, **short}), flush=True)
        long = work / "two-windows.wav"
        repeated = np.concatenate((audio, np.zeros(CHUNK_SAMPLES - len(audio), dtype=np.float32), audio))
        sf.write(long, repeated, SAMPLE_RATE, subtype="FLOAT")
        double = native(args, long, weights, env)
        compare(double, ids, expected, 2)
        print(json.dumps({"case": "two-windows", "backend": args.backend, **double}), flush=True)


if __name__ == "__main__":
    main()
