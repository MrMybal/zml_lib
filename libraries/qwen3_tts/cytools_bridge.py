"""Local Qwen3-TTS launcher adapter for the CyTools speech workers."""

import json
import os
from pathlib import Path
import subprocess
import sys
import wave


PROFILES = {"0b6": ("qwen3_tts_0b6", "0.6B"), "1b7": ("qwen3_tts_1b7", "1.7B")}
MODEL_TYPES = {"custom": ("custom_voice", "CustomVoice"), "design": ("voice_design", "VoiceDesign"),
               "clone": ("base", "Base")}
ENVIRONMENT = {
    "PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "LD_LIBRARY_PATH",
    "CUDA_PATH", "CUDA_HOME", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUDA_CACHE_DISABLE",
    "NVIDIA_TF32_OVERRIDE", "XLA_FLAGS", "ZML_CPU_PJRT_DLL",
}


def local_path(value, label, *, directory=False):
    path = Path(value)
    if not path.is_absolute() or not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"{label} must name an existing absolute local path")
    return path.resolve(strict=True)


def synthesize(config, request, item, output):
    """Run one passage, with operator-configured binaries and converted weights."""
    mode = request["mode"]
    device = request.get("device", "cuda")
    if device not in ("cpu", "cuda"):
        raise ValueError("Qwen3-TTS ZML accepts CPU or CUDA")
    if request.get("precision", "auto") not in ("auto", "float32"):
        raise ValueError("Qwen3-TTS ZML computes in float32")
    placement = config["devices"][device]
    for key in ("device", "pjrt"):
        if not isinstance(placement[key], str) or not placement[key].strip():
            raise ValueError(f"Set devices.{device}.{key} in the ZML configuration")
    original = local_path(request["model_path"], "model", directory=True)
    model = json.loads((original / "config.json").read_text(encoding="utf-8-sig"))
    profile, size = PROFILES[model["tts_model_size"]]
    model_type, family = MODEL_TYPES[mode]
    if model["tts_model_type"] != model_type:
        raise ValueError(f"Mode {mode} requires {family} weights")
    variant = f"{size}-{family}"
    prepared = local_path(config["prepared"][variant], f"prepared.{variant}", directory=True)
    stages = ("text", "codec", "talker", "predictor")
    if mode == "clone":
        stages += ("speaker",)
    for stage in stages:
        local_path(prepared / (stage + ".safetensors"), stage)
    decoder = local_path(config["waveform_weights"], "waveform_weights")
    probe = local_path(config["probe"], "probe")
    interpreter = local_path(config.get("python", sys.executable), "python")
    workspace = local_path(request["workspace"], "workspace", directory=True)
    output = output.resolve()
    if not output.is_relative_to(workspace):
        raise ValueError("The WAV destination must be inside the voice workspace")
    scratch = output.parent
    if not scratch.is_dir() or scratch.drive.upper() == "C:":
        raise ValueError("Place the ZML voice workspace outside C:")
    limit = min(1024, int(request.get("max_new_tokens", 2048)))
    language = item.get("language") or request.get("language") or "auto"
    command = [
        interpreter, "-B", Path(__file__).resolve().with_name("generate.py"),
        "--probe", probe, "--prepared", prepared, "--profile", profile, "--model", original,
        "--backend", device, "--expected-device", placement["device"], "--expected-pjrt", placement["pjrt"],
        "--scratch", scratch, "--waveform-weights", decoder, "--output", output,
        "--text", item["text"], "--language", language, "--max-frames", str(limit),
        "--seed", str(int(request.get("seed", 42)) & 0x7fffffff),
        "--temperature", str(request.get("temperature", 0.9)), "--top-k", str(request.get("top_k", 50)),
        "--top-p", str(request.get("top_p", 1.0)), "--repetition-penalty", str(request.get("repetition_penalty", 1.05)),
    ]
    expected_mode = None
    if mode == "clone":
        fields = {**request, **item}
        reference = local_path(fields["reference_audio"], "reference recording")
        command.extend(("--reference-audio", reference))
        expected_mode = "x_vector_only"
        if not fields.get("x_vector_only"):
            transcript = (fields.get("reference_text") or "").strip()
            if not transcript:
                raise ValueError("Provide the reference transcription for ICL")
            encoder = local_path(config["audio_encoder_weights"], "audio_encoder_weights")
            command.extend(("--reference-text", transcript, "--audio-encoder-weights", encoder))
            expected_mode = "icl"
    else:
        if mode == "custom":
            command.extend(("--speaker", item.get("speaker") or request.get("speaker") or "Ryan"))
        instruction = item.get("instruction", request.get("instruction", ""))
        if instruction and (mode != "custom" or size != "0.6B"):
            command.extend(("--instruction", instruction))
    env = {key: value for key, value in os.environ.items() if key.upper() in ENVIRONMENT}
    env.update({key: str(scratch) for key in ("TMP", "TEMP", "TMPDIR", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR")})
    env["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                             env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if process.returncode:
        raise RuntimeError(f"Qwen3-TTS ZML exited with status {process.returncode}\n{process.stderr[-5000:]}")
    result = json.loads(process.stdout)
    if result["code_backend"] != device or result["waveform_backend"] != "zml":
        raise RuntimeError("Qwen3-TTS returned an unexpected backend")
    if expected_mode and result.get("voice_mode") != expected_mode:
        raise RuntimeError("Qwen3-TTS returned an unexpected voice mode")
    if result["eos"] is not True:
        raise RuntimeError(f"Qwen3-TTS reached its frame budget ({result['frames']} frames)")
    with wave.open(str(output), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getnframes()) != (
            1, 2, 24000, result["samples"],
        ) or audio.getnframes() <= 0:
            raise RuntimeError("Qwen3-TTS returned an invalid PCM waveform")
    return result
