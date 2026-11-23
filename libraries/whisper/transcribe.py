"""Transcribe local audio with a persistent Whisper ZML worker."""

import argparse
from contextlib import contextmanager, redirect_stdout
import json
from math import gcd
import os
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch
from transformers import GenerationConfig, WhisperProcessor
from transformers.generation.logits_process import (
    LogitsProcessorList,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    WhisperTimeStampLogitsProcessor,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build import scratch_directory
from probe_placement import PLATFORMS, verify_placement


SAMPLE_RATE = 16_000
TIMESTAMP_SAMPLES = 320
CHUNK_SAMPLES = 30 * SAMPLE_RATE
MAX_POSITIONS = 448
PROFILES = {"tiny": (80, 3000, 51_865), "large_v3_turbo": (128, 3000, 51_866)}


class Worker:
    def __init__(self, process, capacity, mels, frames, vocab):
        self.process = process
        self.capacity = capacity
        self.mels = mels
        self.frames = frames
        self.vocab = vocab
        self.windows = 0
        self.steps = 0

    def encode(self, features):
        features = np.asarray(features, dtype="<f4", order="C")
        if features.shape != (1, self.mels, self.frames) or not np.isfinite(features).all():
            raise ValueError("Expected a finite Whisper log-mel tensor")
        self.process.stdin.write(b"\0")
        self.process.stdin.write(memoryview(features).cast("B"))
        self.process.stdin.flush()
        if self.process.stdout.read(1) != b"\0":
            raise RuntimeError("Native Whisper worker did not acknowledge the encoder window")
        self.windows += 1

    def scores(self, tokens):
        if not 1 <= len(tokens) <= self.capacity:
            raise ValueError("Whisper token prefix exceeds the decoder capacity")
        ids = np.zeros(self.capacity, dtype="<u4")
        ids[:len(tokens)] = tokens
        self.process.stdin.write(struct.pack("<BI", 1, len(tokens) - 1))
        self.process.stdin.write(memoryview(ids).cast("B"))
        self.process.stdin.flush()
        data = self.process.stdout.read(self.vocab * 4)
        if len(data) != self.vocab * 4:
            raise RuntimeError("Incomplete logits from the native Whisper worker")
        logits = np.frombuffer(data, dtype="<f4").copy()
        if not np.isfinite(logits).all():
            raise ValueError("Native Whisper logits contain nonfinite values")
        self.steps += 1
        return logits


@contextmanager
def open_worker(probe, profile, weights, backend, capacity, work, env, expected_device=None):
    mels, frames, vocab = (8, 32, 96) if profile == "fixture" else PROFILES[profile]
    command = [probe, profile, "transcribe", backend, weights / "encoder.safetensors",
               weights / "decoder.safetensors", str(capacity)]
    log = work / "whisper-worker.log"
    with log.open("xb") as stderr:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, env=env)
        worker = Worker(process, capacity, mels, frames, vocab)
        try:
            yield worker
            process.stdin.close()
            status = process.wait()
        except BaseException:
            if process.poll() is None:
                process.kill()
                process.wait()
            stderr.close()
            sys.stderr.write(log.read_text(encoding="utf-8", errors="replace")[-8000:])
            raise
        finally:
            process.stdin.close()
            process.stdout.close()
    diagnostics = log.read_text(encoding="utf-8", errors="replace")
    if status:
        sys.stderr.write(diagnostics[-8000:])
        raise RuntimeError(f"Native Whisper worker exited with status {status}")
    worker.placement = verify_placement(diagnostics, backend, 2 * worker.windows + worker.steps, expected_device)
    print(f"Whisper {PLATFORMS[backend]}: {worker.windows} encoder windows, {worker.steps} decoder steps", file=sys.stderr)


def read_audio(path):
    from scipy.signal import resample_poly
    import soundfile as sf

    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if not len(audio) or not np.isfinite(audio).all() or rate <= 0:
        raise ValueError("Expected a nonempty audio file with finite samples")
    mono = audio.mean(axis=1, dtype=np.float32)
    if rate != SAMPLE_RATE:
        divisor = gcd(rate, SAMPLE_RATE)
        mono = resample_poly(mono, SAMPLE_RATE // divisor, rate // divisor)
    return mono


def decoder_prompt(processor, config, language):
    prefix = processor.get_decoder_prompt_ids(language=language, task="transcribe", no_timestamps=False)
    return [config.decoder_start_token_id, *(token for _, token in prefix)]


def detect_language(worker, config):
    if not getattr(config, "lang_to_id", None):
        raise ValueError("Automatic language detection requires a multilingual Whisper model")
    scores = worker.scores([config.decoder_start_token_id])
    language = max(config.lang_to_id, key=lambda name: scores[config.lang_to_id[name]])
    return language.removeprefix("<|").removesuffix("|>")


def generate(worker, config, prefix, limit):
    processors = LogitsProcessorList()
    if config.begin_suppress_tokens:
        processors.append(SuppressTokensAtBeginLogitsProcessor(config.begin_suppress_tokens, len(prefix)))
    if config.suppress_tokens:
        processors.append(SuppressTokensLogitsProcessor(config.suppress_tokens))
    processors.append(WhisperTimeStampLogitsProcessor(config, begin_index=len(prefix)))
    tokens = prefix.copy()
    for _ in range(limit):
        logits = torch.from_numpy(worker.scores(tokens))[None]
        scores = processors(torch.tensor([tokens]), logits)[0]
        if not torch.isfinite(scores).any():
            raise RuntimeError("Whisper has no valid next token")
        token = int(scores.argmax())
        if token == config.eos_token_id:
            return tokens[len(prefix):], True
        tokens.append(token)
    return tokens[len(prefix):], False


def segment_tokens(tokens, tokenizer, timestamp_begin, size, *, final_window):
    """Return local sample ranges and the number of source samples to advance."""
    spans = []
    start = None
    text_ids = []
    last_end = 0
    for token in tokens:
        if token < timestamp_begin:
            text_ids.append(token)
            continue
        stamp = min((token - timestamp_begin) * TIMESTAMP_SAMPLES, size)
        if text_ids:
            text = tokenizer.decode(text_ids, skip_special_tokens=True).strip()
            if text:
                spans.append((start if start is not None else last_end, stamp, text))
            last_end = stamp
            text_ids.clear()
            start = None
        else:
            start = stamp
    if text_ids or start is not None:
        if not final_window and spans and 0 < last_end < size:
            # Decode the unfinished part again with the following audio.
            return spans, last_end
        text = tokenizer.decode(text_ids, skip_special_tokens=True).strip()
        if text:
            spans.append((start if start is not None else last_end, size, text))
    return spans, size


def transcribe(audio, processor, config, language, limit, worker):
    prefix = None if language == "auto" else decoder_prompt(processor, config, language)
    seek = 0
    segments = []
    token_windows = []
    complete = True
    while seek < len(audio):
        window = audio[seek:seek + CHUNK_SAMPLES]
        features = processor.feature_extractor(window, sampling_rate=SAMPLE_RATE, return_tensors="np").input_features
        worker.encode(features)
        if prefix is None:
            language = detect_language(worker, config)
            prefix = decoder_prompt(processor, config, language)
        tokens, finished = generate(worker, config, prefix, limit)
        token_windows.append(tokens)
        complete &= finished
        spans, advance = segment_tokens(tokens, processor.tokenizer, config.no_timestamps_token_id + 1,
                                        len(window), final_window=seek + len(window) == len(audio))
        for start, end, text in spans:
            segments.append({"start": (seek + start) / SAMPLE_RATE, "end": (seek + end) / SAMPLE_RATE, "text": text})
        seek += advance
    return {"text": " ".join(seg["text"] for seg in segments), "segments": segments, "language": language,
            "duration": len(audio) / SAMPLE_RATE, "complete": complete, "token_windows": token_windows}


@contextmanager
def work_directory(parent):
    if parent is None:
        with scratch_directory() as work:
            yield work
        return
    parent = parent.resolve(strict=True)
    with TemporaryDirectory(prefix="whisper-", dir=parent) as work:
        yield Path(work)


def load_runtime_config(path, backend):
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("Expected a Whisper runtime configuration object")
    probe = Path(config["probe"])
    weights = Path(config["weights"])
    device = config.get("expected_cuda_device") if backend == "cuda" else None
    if not probe.is_absolute() or not weights.is_absolute():
        raise ValueError("Whisper probe and weights paths must be absolute")
    if device is not None and not isinstance(device, str):
        raise ValueError("expected_cuda_device must be a device name or substring")
    return probe, weights, device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path, help="local Transformers tokenizer and configuration")
    parser.add_argument("--weights", type=Path, help="directory from prepare.py")
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--runtime-config", type=Path, help="local CyTools Whisper runtime JSON")
    parser.add_argument("--profile", choices=PROFILES, default="large_v3_turbo")
    parser.add_argument("--language", required=True, help="Whisper language code such as fr, en or auto")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="limit per 30-second window")
    parser.add_argument("--backend", choices=PLATFORMS, default="cpu")
    parser.add_argument("--expected-device")
    parser.add_argument("--workspace", type=Path, help="existing directory for job-owned scratch files")
    args = parser.parse_args()
    if args.runtime_config is not None:
        if args.weights is not None or args.probe is not None or args.expected_device is not None:
            parser.error("Use --runtime-config or the explicit probe, weights and device paths")
        try:
            args.probe, args.weights, args.expected_device = load_runtime_config(args.runtime_config, args.backend)
        except (OSError, KeyError, TypeError, ValueError) as error:
            parser.error(f"Invalid Whisper runtime configuration: {error}")
    elif args.weights is None or args.probe is None:
        parser.error("Specify --probe and --weights or pass --runtime-config")
    model = args.model.resolve(strict=True)
    config = GenerationConfig.from_pretrained(model, local_files_only=True)
    processor = WhisperProcessor.from_pretrained(model, local_files_only=True)
    # The multilingual prompt has one language token, regardless of the selected language.
    prompt_language = "en" if args.language == "auto" else args.language
    prefix = decoder_prompt(processor, config, prompt_language)
    capacity = len(prefix) + args.max_new_tokens
    if not 1 <= args.max_new_tokens or capacity > MAX_POSITIONS:
        parser.error("The prompt and requested tokens must fit 448 decoder positions")
    if processor.feature_extractor.feature_size != PROFILES[args.profile][0]:
        parser.error("The processor log-mel dimensions do not match the native profile")
    torch.set_num_threads(1)
    audio = read_audio(args.audio)
    probe, weights = (path.resolve(strict=True) for path in (args.probe, args.weights))
    with redirect_stdout(sys.stderr), work_directory(args.workspace) as work:
        env = os.environ.copy()
        env.update({key: str(work) for key in ("TEMP", "TMP", "TMPDIR")})
        env["CUDA_CACHE_DISABLE"] = "1"
        with open_worker(probe, args.profile, weights, args.backend, capacity, work, env, args.expected_device) as worker:
            report = transcribe(audio, processor, config, args.language, args.max_new_tokens, worker)
        report["placement"] = worker.placement
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
