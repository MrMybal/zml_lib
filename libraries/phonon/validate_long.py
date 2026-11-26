"""Compare long Phonon transcription with independent inference for each segment."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import soundfile

from transcribe import MAX_SECONDS, format_window, invoke, load_tokenizer, log_mels, recording_windows


def validate(args, work):
    command = [sys.executable, "-B", str(Path(__file__).with_name("transcribe.py")),
               "--probe", args.probe, "--encoder", args.encoder, "--tdt", args.tdt,
               "--tokenizer", args.tokenizer, "--audio", args.audio, "--scratch", args.scratch,
               "--backend", args.backend, "--expected-device", args.expected_device,
               "--chunk-seconds", str(args.chunk_seconds), "--overlap-seconds", str(args.overlap_seconds)]
    if args.expected_pjrt:
        command.extend(("--expected-pjrt", args.expected_pjrt))
    process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if process.returncode:
        raise RuntimeError(process.stderr[-7000:])
    observed = json.loads(process.stdout)
    if len(observed.get("segments", ())) < 2:
        raise ValueError("Expected an audio file longer than one segment")

    tokenizer, special = load_tokenizer(args.tokenizer)
    expected = {"tokens": [], "words": [], "segments": []}
    with soundfile.SoundFile(args.audio) as audio:
        expected["seconds"] = round(len(audio) / audio.samplerate, 4)
        windows = recording_windows(audio, args.chunk_seconds, args.overlap_seconds)
        for index, (data, start, end, left, right) in enumerate(windows):
            segment = log_mels(data)
            features, trace = work / f"{index}.f32", work / f"{index}.json"
            segment.astype("<f4", copy=False).tofile(features)
            invoke(args.probe, args.backend, args.expected_device, args.expected_pjrt, "transcribe",
                   args.encoder, args.tdt, str(len(segment)), features, trace)
            decoded, window = format_window(json.loads(trace.read_text(encoding="utf-8")), tokenizer, special,
                                            start, end, left, right, args.overlap_seconds)
            expected["tokens"].extend(decoded["tokens"])
            expected["words"].extend(decoded["words"])
            expected["segments"].append(window)
            features.unlink()
            trace.unlink()

    expected["text"] = " ".join(segment["text"] for segment in expected["segments"] if segment["text"])
    for key, value in expected.items():
        if observed[key] != value:
            raise ValueError(f"The persistent worker changed {key} compared to independent segments")
    breaks = observed["segments"]
    if breaks[0]["start"] != 0 or breaks[-1]["end"] != observed["seconds"] or any(
        left["end"] != right["start"] for left, right in zip(breaks, breaks[1:])
    ):
        raise ValueError("Audio segments did not cover the full recording")
    for kind in ("words", "tokens"):
        current = observed[kind]
        if not current or any(not 0 <= item["start"] <= item["end"] <= observed["seconds"] for item in current):
            raise ValueError(f"Invalid {kind} timestamps")
        if not args.overlap_seconds and any(left["start"] > right["start"] for left, right in zip(current, current[1:])):
            raise ValueError(f"Disordered {kind} timestamps")

    if args.output:
        with args.output.open("x", encoding="utf-8") as destination:
            json.dump(observed, destination, ensure_ascii=True)
            destination.write("\n")
    print(json.dumps({"seconds": observed["seconds"], "segments": len(breaks),
                      "ends": [row["end"] for row in breaks], "words": len(observed["words"]),
                      "tokens": len(observed["tokens"]), "placement": observed["placement"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("probe", "encoder", "tdt", "audio", "tokenizer", "scratch"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--expected-device", required=True)
    parser.add_argument("--expected-pjrt")
    parser.add_argument("--chunk-seconds", type=float, default=MAX_SECONDS)
    parser.add_argument("--overlap-seconds", type=float, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.scratch = args.scratch.resolve(strict=True)
    if args.scratch.drive.upper() == "C:":
        parser.error("--scratch must be outside C:")
    with TemporaryDirectory(prefix="phonon-long-validation-", dir=args.scratch) as work:
        validate(args, Path(work))


if __name__ == "__main__":
    main()
