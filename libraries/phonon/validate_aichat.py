"""Compare the public CyTools AIChat CLI with a direct Phonon ZML transcription."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def run_cli(args, work, environment):
    config = work / "request.json"
    config.write_text(json.dumps({
        "audio": str(args.audio), "model": args.model, "asr_engine": "zml",
    }), encoding="utf-8")
    reply = subprocess.run(
        [args.python, "-B", args.tool, "--data-dir", work / "jobs", "invoke", "transcribe",
         "--params-file", config, "--timeout", str(args.timeout)],
        env=environment, capture_output=True, text=True, encoding="utf-8", errors="strict",
        timeout=args.timeout + 30,
    )
    if reply.returncode:
        raise RuntimeError(f"AIChat exited with status {reply.returncode}\n"
                           f"{reply.stdout[-4000:]}\n{reply.stderr[-4000:]}")
    job = json.loads(reply.stdout)
    if job["operationId"] != "transcribe" or job["state"] != "Completed":
        raise ValueError(f"AIChat did not complete the transcription: {job.get('error')}")
    if job["artifactsDeleted"] is not True:
        raise ValueError("AIChat did not finish its temporary task")
    if any((work / "jobs").rglob("phonon-*")):
        raise ValueError("AIChat left a Phonon scratch directory")
    return job


def validate(args, work):
    reference = read_json(args.reference)
    config = read_json(args.config)
    placement = reference["placement"]
    if (placement["backend"], placement["pjrt"], placement["shards"]) != (
        config["backend"], config["expected_pjrt"], 1,
    ) or config["expected_device"].casefold() not in placement["device"].casefold():
        raise ValueError("The direct run used a different device than the AIChat configuration")
    if not reference["text"].strip() or not 0 < reference["seconds"] < math.inf:
        raise ValueError("Use a direct transcription of a nonempty speech recording")

    audio_sha = digest(args.audio)
    if args.audio_sha256 and audio_sha != args.audio_sha256.lower():
        raise ValueError("The source recording does not match the expected SHA-256")
    environment = os.environ.copy()
    environment.pop("CYTOOLS_TOKEN", None)
    environment.update({name: str(work) for name in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME")})
    environment.update(CYTOOLS_PHONON_ZML_CONFIG=str(args.config), PYTHONDONTWRITEBYTECODE="1",
                       PYTHONIOENCODING="utf-8")

    runs = []
    for _ in range(args.repeats):
        job = run_cli(args, work, environment)
        result = job["outputs"]
        expected = (args.model, "zml", reference["text"].strip(), placement)
        actual = (result["model"], result["asrEngine"], result["text"], result["placement"])
        if actual != expected or not math.isclose(result["audioSeconds"], reference["seconds"],
                                                  rel_tol=0, abs_tol=0.0001):
            raise ValueError(f"AIChat and direct ZML differ\nDirect {expected!r}\nAIChat {actual!r}")
        runs.append({"jobId": job["jobId"], "taskCleaned": job["artifactsDeleted"], **result})

    if len({run["jobId"] for run in runs}) != len(runs) or digest(args.audio) != audio_sha:
        raise ValueError("The repeated jobs or the source recording changed unexpectedly")
    return {"audioSha256": audio_sha, "direct": reference, "runs": runs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("tool", "config", "reference", "audio", "scratch", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--model", choices=("phonon-2", "phonon-2.5", "phonon-2.5-en"), required=True)
    parser.add_argument("--audio-sha256", help="expected digest of the input used for the direct run")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=600, help="timeout per job in seconds")
    args = parser.parse_args()
    if args.repeats < 1 or args.timeout < 1:
        parser.error("repeats and timeout must be positive")
    for name in ("tool", "python", "config", "reference", "audio"):
        path = getattr(args, name).resolve(strict=True)
        if not path.is_file():
            parser.error(f"--{name} must be a file")
        setattr(args, name, path)
    args.scratch = args.scratch.resolve(strict=True)
    if not args.scratch.is_dir() or args.scratch.drive.upper() == "C:":
        parser.error("--scratch must be an existing directory outside C:")
    if args.report.exists():
        parser.error("--report already exists")
    with TemporaryDirectory(prefix="phonon-aichat-", dir=args.scratch) as temporary:
        result = validate(args, Path(temporary))
    with args.report.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print(json.dumps({
        "matches": len(result["runs"]), "model": args.model,
        "audioSeconds": result["direct"]["seconds"], "placement": result["direct"]["placement"],
    }))


if __name__ == "__main__":
    main()
