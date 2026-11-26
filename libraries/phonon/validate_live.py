"""Validate Phonon live windows against independent, unpadded inference."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

import numpy as np
import soundfile

from transcribe import HOP, LIVE_SECONDS, RATE, invoke, log_mels, read_audio


def launcher(args, work):
    command = [sys.executable, "-B", Path(__file__).with_name("transcribe.py"),
               "--probe", args.probe, "--encoder", args.encoder, "--tdt", args.tdt,
               "--tokenizer", args.tokenizer, "--scratch", work, "--backend", args.backend]
    if args.expected_pjrt:
        command.extend(("--expected-pjrt", args.expected_pjrt))
    return command


def check_encoder(args, audio, work):
    waveform, _ = read_audio(audio)
    features = log_mels(waveform)
    inputs, fixed, reference = (work / f"{audio.stem}-{name}.f32" for name in ("input", "fixed", "reference"))
    features.astype("<f4", copy=False).tofile(inputs)
    settings = (args.probe, args.backend, args.expected_device, args.expected_pjrt)
    invoke(*settings, "encode", args.encoder, str(len(features)), inputs, reference)
    invoke(*settings, "encode-live", args.encoder, str(len(features)), inputs, fixed)
    expected = np.fromfile(reference, dtype="<f4")
    actual = np.fromfile(fixed, dtype="<f4")
    if (expected.shape != actual.shape or not expected.size
            or not np.isfinite(expected).all() or not np.isfinite(actual).all()):
        raise ValueError("Fixed encoder returned invalid features")
    error = float(np.sqrt(np.mean((actual.astype(np.float64) - expected) ** 2)))
    if error > 0.001:
        raise ValueError(f"Fixed encoder differs for {len(features)} input frames (RMSE {error})")
    return {"frames": len(features), "rmse": error}


def check_stream(args, recordings, work):
    command = launcher(args, work)
    device = ["--expected-device", args.expected_device]
    windows = []
    with (work / "launcher.log").open("w+b") as stderr, subprocess.Popen(
        [*command, *device, "--live"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=stderr, text=True, encoding="utf-8",
    ) as child:
        try:
            line = child.stdout.readline()
            if not line:
                stderr.seek(0)
                raise RuntimeError(stderr.read().decode("utf-8", errors="replace")[-7000:])
            ready = json.loads(line)
            if ready["event"] != "ready" or ready["loadSeconds"] < 0:
                raise ValueError(f"Invalid readiness report {ready}")
            for path in recordings:
                started = time.monotonic()
                child.stdin.write(json.dumps({"audio": str(path)}) + "\n")
                child.stdin.flush()
                window = json.loads(child.stdout.readline())
                elapsed = time.monotonic() - started
                if window.pop("event") != "transcription":
                    raise ValueError("Invalid live transcription event")
                if args.max_response_seconds is not None and elapsed > args.max_response_seconds:
                    raise ValueError(f"Live window took {elapsed:.3f} seconds after warmup")
                windows.append({"source": path.name, "responseSeconds": round(elapsed, 3), "transcript": window})
            child.stdin.close()
            if json.loads(child.stdout.readline()) != {"event": "closed", "windows": len(windows)}:
                raise ValueError("The launcher did not confirm completion of all windows")
            if child.wait() != 0:
                raise RuntimeError("The live launcher exited with an error")
        except BaseException:
            child.kill()
            raise

    references = {}
    for audio in dict.fromkeys(recordings):
        direct = subprocess.run([*command, *device, "--audio", audio], capture_output=True, check=True)
        references[audio.name] = json.loads(direct.stdout)
    for window in windows:
        if window["transcript"] != references[window["source"]]:
            raise ValueError(f"Live tokens or timestamps differ for {window['source']}")
    if not any(window["transcript"]["words"] for window in windows):
        raise ValueError("The sample did not contain recognized speech")

    wrong = subprocess.run([*command, "--expected-device", "phonon-test-unavailable", "--live"],
                           capture_output=True)
    if not wrong.returncode or wrong.stdout.strip() or b"Expected device matching" not in wrong.stderr:
        raise ValueError("An incorrect device was accepted")
    return {"ready": ready, "windows": windows, "matchesIndependentRuns": True, "badPlacementRejected": True,
            "nativeExited": True}


def validate(args, work):
    waveform, seconds = read_audio(args.audio)
    if not 1.25 < seconds <= LIVE_SECONDS:
        raise ValueError(f"Use a speech recording longer than 1.25 seconds and at most {LIVE_SECONDS} seconds")
    # Exercise both padding parities at the first convolution.
    odd = work / "odd-121.wav"
    even = work / "even-122.wav"
    soundfile.write(odd, waveform[:121 * HOP], RATE, subtype="FLOAT")
    soundfile.write(even, waveform[:122 * HOP], RATE, subtype="FLOAT")
    result = check_stream(args, (odd, args.audio, even, odd), work)
    result["encoder"] = [check_encoder(args, path, work) for path in (odd, even)]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("probe", "encoder", "tdt", "tokenizer", "audio", "scratch"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "rocm", "vulkan"), required=True)
    parser.add_argument("--expected-device", required=True)
    parser.add_argument("--expected-pjrt")
    parser.add_argument("--max-response-seconds", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.scratch = args.scratch.resolve(strict=True)
    if args.scratch.drive.upper() == "C:":
        parser.error("--scratch must be outside C:")
    with TemporaryDirectory(prefix="phonon-live-validation-", dir=args.scratch) as folder:
        result = validate(args, Path(folder))
    if args.output:
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2, ensure_ascii=True)
            output.write("\n")
    print(json.dumps({"warmupSeconds": result["ready"]["loadSeconds"], "encoder": result["encoder"],
                      "responseSeconds": [window["responseSeconds"] for window in result["windows"]]}))


if __name__ == "__main__":
    main()
