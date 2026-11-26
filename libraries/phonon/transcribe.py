"""Transcribe a recording with the ZML Phonon FastConformer and TDT."""

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

import librosa
import numpy as np
from scipy.fft import rfft
from scipy.signal import resample_poly
import soundfile
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.probe_placement import verify_placement

RATE = 16000
HOP = 160
FFT = 512
WINDOW = 400
MELS = 128
FRAME_TIME = 8 * HOP / RATE
BLANK = 8192
MAX_SECONDS = 30
LIVE_SECONDS = 20
MIN_SECONDS = 0.02
MODEL_READY = "phonon_stream_models_loaded"


def resample(samples, source_rate):
    if not np.isfinite(samples).all():
        raise ValueError("The input audio contains nonfinite samples")
    if source_rate != RATE:
        divisor = math.gcd(source_rate, RATE)
        samples = resample_poly(samples, RATE // divisor, source_rate // divisor)
    return samples


def read_audio(path):
    with soundfile.SoundFile(path) as stream:
        seconds = len(stream) / stream.samplerate
        if not MIN_SECONDS <= seconds <= MAX_SECONDS:
            raise ValueError(f"Expected one audio segment between {MIN_SECONDS} and {MAX_SECONDS} seconds")
        samples = resample(stream.read(dtype="float32", always_2d=True).mean(axis=1), stream.samplerate)
    return samples, seconds


def cut_near_silence(samples, rate, limit):
    step = max(1, round(rate * FRAME_TIME))
    radius = step
    search = min(4 * rate, len(samples) // 5)
    choices = np.arange(((limit - search + step - 1) // step) * step,
                        min(limit, len(samples) - radius) + 1, step)
    square = np.square(samples, dtype=np.float64)
    sums = np.concatenate(([0.0], np.cumsum(square)))
    energies = (sums[choices + radius] - sums[choices - radius]) / (2 * radius)
    quiet = choices[energies <= max(0.003**2, 0.15**2 * square.mean())]
    return int(quiet[-1]) if len(quiet) else limit


def audio_chunks(stream, chunk_seconds):
    rate, total = stream.samplerate, len(stream)
    window = round(chunk_seconds * rate)
    minimum = math.ceil(MIN_SECONDS * rate)
    start = 0
    while start < total:
        count = min(window, total - start)
        stream.seek(start)
        samples = stream.read(count, dtype="float32", always_2d=True).mean(axis=1)
        if len(samples) != count:
            raise ValueError(f"The audio ended unexpectedly at sample {start + len(samples)}")
        if not np.isfinite(samples).all():
            raise ValueError("The input audio contains nonfinite samples")
        if start + count < total:
            count = cut_near_silence(samples, rate, min(count, total - start - minimum))
        end = start + count
        yield resample(samples[:count], rate), start / rate, end / rate
        start = end


def recording_windows(stream, chunk_seconds, overlap_seconds):
    if not overlap_seconds:
        for samples, start, end in audio_chunks(stream, chunk_seconds):
            yield samples, start, end, start, end
        return
    rate, total = stream.samplerate, len(stream)
    index, start = 0, 0
    while start < total:
        end = min(total, round((index + 1) * chunk_seconds * rate))
        left = max(0, round((index * chunk_seconds - overlap_seconds / 2) * rate))
        right = min(total, round(((index + 1) * chunk_seconds + overlap_seconds / 2) * rate))
        stream.seek(left)
        samples = stream.read(right - left, dtype="float32", always_2d=True).mean(axis=1)
        if len(samples) != right - left:
            raise ValueError(f"The audio ended unexpectedly at sample {left + len(samples)}")
        yield resample(samples, rate), start / rate, end / rate, left / rate, right / rate
        index, start = index + 1, end


def log_mels(waveform):
    emphasized = np.concatenate((waveform[:1], waveform[1:] - np.float32(0.97) * waveform[:-1]))
    centered = np.pad(emphasized, FFT // 2)
    blocks = np.lib.stride_tricks.sliding_window_view(centered, FFT)[::HOP][:len(waveform) // HOP]
    margin = (FFT - WINDOW) // 2
    window = np.pad(np.hanning(WINDOW).astype(np.float32), margin)
    spectrum = np.abs(rfft(blocks * window, axis=-1))**2
    filters = librosa.filters.mel(sr=RATE, n_fft=FFT, n_mels=MELS, norm="slaney", dtype=np.float32)
    features = np.log(spectrum @ filters.T + np.float32(2**-24))
    return (features - features.mean(axis=0)) / (features.std(axis=0, ddof=1) + np.float32(1.e-5))


def native_env(backend):
    env = os.environ.copy()
    if backend == "cuda" and sys.platform == "linux":
        env.setdefault("NVIDIA_TF32_OVERRIDE", "0")
        env.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false --xla_gpu_enable_cublaslt=false")
    return env


def invoke(probe, backend, device, pjrt, mode, *parameters):
    process = subprocess.run([probe, mode, backend, *parameters], capture_output=True, text=True, encoding="utf-8",
                             errors="replace", env=native_env(backend))
    if process.returncode:
        raise RuntimeError(process.stderr[-7000:] or f"Phonon exited with status {process.returncode}")
    return verify_placement(process.stderr, backend, 2 if mode == "transcribe" else 1, device, expected_pjrt=pjrt)


def load_tokenizer(tokenizer_path):
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    metadata = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    special = {entry["id"] for entry in metadata.get("added_tokens", ()) if entry["special"]}
    return tokenizer, special


def format_events(events, tokenizer, special, seconds, start=0.0, ownership=None):
    tokens, groups = [], []
    ids = []
    for row in events:
        index = row["token"]
        if index == BLANK or index in special:
            continue
        piece = tokenizer.id_to_token(index)
        if piece is None:
            raise ValueError(f"The tokenizer is missing token {index}")
        ids.append(index)
        time = start + min(row["frame"] * FRAME_TIME, seconds)
        end = start + min((row["frame"] + max(1, row["duration"])) * FRAME_TIME, seconds)
        token = {"id": index, "start": round(time, 4), "end": round(end, 4)}
        if not groups or piece.startswith(("▁", " ")):
            groups.append(([], token["start"]))
        groups[-1][0].append(index)
        tokens.append(token)
    words, owned_tokens, offset = [], [], 0
    for indices, start in groups:
        first = offset
        offset += len(indices)
        word = tokenizer.decode(indices).strip()
        end = tokens[offset - 1]["end"]
        if offset < len(tokens):
            end = min(end, tokens[offset]["start"])
        if ownership is not None:
            center = (start + end) / 2
            if not ownership[0] <= center < ownership[1]:
                continue
            owned_tokens.extend(tokens[first:offset])
        if word:
            words.append({"text": word, "start": start, "end": end})
    text = tokenizer.decode(ids).strip() if ownership is None else " ".join(word["text"] for word in words)
    return {"text": text, "words": words, "tokens": tokens if ownership is None else owned_tokens,
            "seconds": round(seconds, 4)}


def format_text(events, tokenizer_path, seconds):
    return format_events(events, *load_tokenizer(tokenizer_path), seconds)


def format_window(events, tokenizer, special, start, end, left, right, overlap):
    ownership = (start, end) if overlap else None
    decoded = format_events(events, tokenizer, special, right - left, left, ownership)
    segment = {"start": round(start, 4), "end": round(end, 4), "text": decoded["text"]}
    if overlap:
        segment.update(window_start=round(left, 4), window_end=round(right, 4))
    return decoded, segment


def transcribe_segment(args, work):
    audio, seconds = read_audio(args.audio)
    mels = log_mels(audio)
    packed, trace = work / "mels.f32", work / "trace.json"
    mels.astype("<f4", copy=False).tofile(packed)
    placement = invoke(args.probe, args.backend, args.expected_device, args.expected_pjrt, "transcribe",
                       args.encoder, args.tdt, str(len(mels)), packed, trace)
    report = format_text(json.loads(trace.read_text(encoding="utf-8")), args.tokenizer, seconds)
    report["placement"] = placement
    return report


@contextmanager
def native_stream(args, work, mode="stream"):
    command = [args.probe, mode, args.backend, args.encoder, args.tdt]
    log = work / "stream.log"
    with log.open("wb") as diagnostics, subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=diagnostics, env=native_env(args.backend),
    ) as process:
        try:
            yield process, log
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("The native Phonon stream exited with an error")
        except BaseException as error:
            process.kill()
            process.wait()
            if isinstance(error, (BrokenPipeError, RuntimeError)):
                reason = log.read_text(encoding="utf-8", errors="replace")[-7000:]
                raise RuntimeError(reason or str(error)) from error
            raise
    if log.read_text(encoding="utf-8", errors="replace").count(MODEL_READY) != 1:
        raise RuntimeError("Phonon did not confirm that both models were loaded once")


def stream_events(process, waveform):
    features = np.ascontiguousarray(log_mels(waveform), dtype="<f4")
    process.stdin.write(struct.pack("<I", len(features)))
    process.stdin.write(features.tobytes())
    process.stdin.flush()
    reply = process.stdout.readline(1 << 20)
    if not reply.endswith(b"\n"):
        raise RuntimeError("Phonon ended a segment before returning its trace")
    return json.loads(reply)


def transcribe_recording(args, work, chunk_seconds, overlap_seconds):
    tokenizer, special = load_tokenizer(args.tokenizer)
    report = {"words": [], "tokens": [], "segments": []}
    with soundfile.SoundFile(args.audio) as audio, native_stream(args, work) as (process, log):
        report["seconds"] = round(len(audio) / audio.samplerate, 4)
        for waveform, start, end, left, right in recording_windows(audio, chunk_seconds, overlap_seconds):
            events = stream_events(process, waveform)
            decoded, segment = format_window(events, tokenizer, special, start, end, left, right, overlap_seconds)
            report["words"].extend(decoded["words"])
            report["tokens"].extend(decoded["tokens"])
            report["segments"].append(segment)
    stderr = log.read_text(encoding="utf-8", errors="replace")
    report["placement"] = verify_placement(stderr, args.backend, 2 * len(report["segments"]),
                                           args.expected_device, expected_pjrt=args.expected_pjrt)
    report["text"] = " ".join(segment["text"] for segment in report["segments"] if segment["text"])
    return report


def transcribe(args, work):
    chunk_seconds = getattr(args, "chunk_seconds", MAX_SECONDS)
    info = soundfile.info(args.audio)
    if info.frames <= chunk_seconds * info.samplerate:
        return transcribe_segment(args, work)
    return transcribe_recording(args, work, chunk_seconds, getattr(args, "overlap_seconds", 0))


def transcribe_live(args, work, requests, responses):
    """Run successive audio windows through the same native Phonon process."""
    def send(**report):
        print(json.dumps(report, ensure_ascii=True), file=responses, flush=True)

    started = time.monotonic()
    tokenizer, special = load_tokenizer(args.tokenizer)
    windows = 0
    with native_stream(args, work, "stream-live") as (process, log), log.open("rb") as diagnostics:
        startup = bytearray()
        while MODEL_READY.encode() not in startup:
            if process.poll() is not None or time.monotonic() - started >= 180:
                raise RuntimeError("Phonon did not load the models")
            startup.extend(diagnostics.read())
            time.sleep(0.025)
        # Compile the fixed encoder and decoder before advertising readiness.
        stream_events(process, np.zeros(RATE, dtype=np.float32))
        verify_placement(diagnostics.read().decode("utf-8", errors="replace"),
                         args.backend, 2, args.expected_device, expected_pjrt=args.expected_pjrt)
        send(event="ready", loadSeconds=round(time.monotonic() - started, 3))
        while line := requests.readline(65536):
            if not line.endswith("\n"):
                raise ValueError("Expected a bounded JSON line containing an audio path")
            request = json.loads(line)
            waveform, seconds = read_audio(Path(request["audio"]))
            if seconds > LIVE_SECONDS:
                raise ValueError(f"Live windows must be at most {LIVE_SECONDS} seconds")
            events = stream_events(process, waveform)
            placement = verify_placement(diagnostics.read().decode("utf-8", errors="replace"),
                                         args.backend, 2, args.expected_device, expected_pjrt=args.expected_pjrt)
            report = format_events(events, tokenizer, special, seconds)
            send(event="transcription", placement=placement, **report)
            windows += 1
    send(event="closed", windows=windows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("probe", "encoder", "tdt", "tokenizer", "scratch"):
        parser.add_argument("--" + flag, type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audio", type=Path)
    mode.add_argument("--live", action="store_true", help="accept audio paths as JSON lines on stdin")
    parser.add_argument("--backend", choices=("cpu", "cuda", "rocm", "vulkan"), required=True)
    parser.add_argument("--expected-device", required=True)
    parser.add_argument("--expected-pjrt")
    parser.add_argument("--chunk-seconds", type=float, default=MAX_SECONDS, metavar="[5-30]",
                        help="duration of each FastConformer segment before overlap")
    parser.add_argument("--overlap-seconds", type=float, default=0, metavar="[0-4]",
                        help="total overlap shared between both sides of a segment")
    args = parser.parse_args()
    if not 5 <= args.chunk_seconds <= MAX_SECONDS or not (args.overlap_seconds == 0 or 0.2 <= args.overlap_seconds <= 4):
        parser.error("use 5–30 seconds per segment and either zero overlap or 0.2–4 seconds")
    if args.chunk_seconds + args.overlap_seconds > MAX_SECONDS:
        parser.error("--chunk-seconds plus --overlap-seconds must not exceed 30 seconds")
    args.scratch = args.scratch.resolve(strict=True)
    if args.scratch.drive.upper() == "C:":
        parser.error("--scratch must be outside C:")
    with TemporaryDirectory(prefix="phonon-asr-", dir=args.scratch) as work:
        if args.live:
            transcribe_live(args, Path(work), sys.stdin, sys.stdout)
        else:
            print(json.dumps(transcribe(args, Path(work)), ensure_ascii=True))


if __name__ == "__main__":
    main()
