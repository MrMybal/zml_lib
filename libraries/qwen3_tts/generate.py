"""Generate codes with ZML Qwen3-TTS and optionally decode them to a WAV."""

import argparse
from contextlib import ExitStack, contextmanager, redirect_stdout
from itertools import groupby
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import wave

import numpy as np
from safetensors.numpy import load_file
from transformers import AutoTokenizer

from prepare import PROFILES, native_env, verify_config, verify_speaker_config
from scripts.probe_placement import verify_placement

WAVEFORM_RATE = 24_000
SAMPLES_PER_FRAME = 1920
WAVEFORM_CHUNK = 300
WAVEFORM_CONTEXT = 25
TALKER_INITIAL_FRAMES = 64


def draw_code(probabilities, random):
    cumulative = np.cumsum(probabilities, dtype=np.float64)
    return int(np.searchsorted(cumulative, random.random() * cumulative[-1], side="right"))


def sample_code(logits, random, temperature=0.9, top_k=50, top_p=1.0):
    scaled = logits / np.float32(temperature)
    if 0 < top_k < len(scaled):
        cutoff = np.partition(scaled, len(scaled) - top_k)[len(scaled) - top_k]
        scaled = np.where(scaled >= cutoff, scaled, -np.inf)
    probabilities = np.exp(scaled - scaled.max())
    if top_p < 1.0:
        order = np.argsort(scaled, kind="stable")
        cumulative = np.cumsum(probabilities[order], dtype=np.float64)
        remove = cumulative <= (1.0 - top_p) * cumulative[-1]
        remove[-1] = False
        probabilities[order[remove]] = 0
    return draw_code(probabilities, random)


@contextmanager
def runner(args, work, stage, rows, output_shape):
    if stage in ("waveform", "audio_encoder"):
        weights = args.waveform_weights if stage == "waveform" else args.audio_encoder_weights
        command = [args.probe, stage, args.backend, weights, str(rows), "stdin"]
    else:
        weights = args.prepared / f"{stage}.safetensors"
        command = [args.probe, args.profile, stage, args.backend, weights, str(rows), "stdin"]
    size = int(np.prod(output_shape)) * 4
    log = work / f"{stage}.log"
    with log.open("wb") as diagnostics, subprocess.Popen(
        command, env=native_env(args.backend), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=diagnostics,
    ) as proc:
        calls = 0

        def infer(values):
            nonlocal calls
            proc.stdin.write(b"\0")
            proc.stdin.write(np.ascontiguousarray(values, dtype="<f4").tobytes())
            proc.stdin.flush()
            response = proc.stdout.read(size)
            if len(response) != size:
                raise RuntimeError(f"{stage} did not return a complete result\n{log.read_text(encoding='utf-8', errors='replace')[-5000:]}")
            calls += 1
            return np.frombuffer(response, dtype="<f4").reshape(output_shape)

        try:
            yield infer
        except BaseException:
            proc.kill()
            raise
        else:
            proc.stdin.close()
            if proc.wait(timeout=60) != 0:
                raise RuntimeError(log.read_text(encoding="utf-8", errors="replace")[-5000:])
            if calls:
                verify_placement(log.read_text(encoding="utf-8", errors="replace"), args.backend, calls,
                                 args.expected_device, expected_pjrt=args.expected_pjrt)
        finally:
            if proc.poll() is None:
                proc.kill()


def encode_reference(args, work, samples=None):
    from speaker import read_reference

    if samples is None:
        samples = read_reference(args.reference_audio)
    frames = (len(samples) + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME
    with runner(args, work, "audio_encoder", len(samples), (frames, 16)) as encode:
        result = encode(samples)
    codes = result.astype(np.int64)
    if not np.array_equal(result, codes) or codes.min() < 0 or codes.max() >= 2048:
        raise ValueError("The audio encoder returned invalid codes")
    return codes


def frame_embeddings(codec, frames):
    return np.sum([table[frames[:, group]].astype(np.float32) for group, table in enumerate(codec)], axis=0)


def prompt(args, work, config, codec, reference_codes=None):
    talker = config["talker_config"]
    profile = config["tts_model_type"]
    if profile == "custom_voice":
        if args.speaker not in talker["spk_id"]:
            raise ValueError(f"Choose one of the speakers {', '.join(talker['spk_id'])}")
        if args.instruction and args.profile == "qwen3_tts_0b6":
            raise ValueError("CustomVoice 0.6B accepts a speaker and language")
    elif profile == "voice_design":
        if not args.instruction or args.speaker:
            raise ValueError("VoiceDesign requires a voice instruction")
    elif profile == "base":
        if args.reference_audio is None or args.speaker or args.instruction:
            raise ValueError("Base requires --reference-audio to clone a voice")
    else:
        raise ValueError("Unsupported Qwen3-TTS model type")
    language = args.language
    dialect = talker["spk_is_dialect"].get(args.speaker)
    if language in ("auto", "chinese") and dialect:
        language = dialect
    if language == "auto":
        tags = [talker["codec_nothink_id"], talker["codec_think_bos_id"], talker["codec_think_eos_id"]]
    else:
        if language not in talker["codec_language_id"]:
            raise ValueError(f"Unknown language {language}")
        tags = [talker["codec_think_id"], talker["codec_think_bos_id"],
                talker["codec_language_id"][language], talker["codec_think_eos_id"]]
    if args.speaker:
        tags.append(talker["spk_id"][args.speaker])
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    message = f"<|im_start|>assistant\n{args.text}<|im_end|>\n<|im_start|>assistant\n"
    ids = tokenizer.encode(message, add_special_tokens=False)
    if len(ids) <= 8:
        raise ValueError("The text produced no content tokens")
    describe = tokenizer.encode(f"<|im_start|>user\n{args.instruction}<|im_end|>\n", add_special_tokens=False) if args.instruction else []
    transcript = []
    reference_frames = 0
    if reference_codes is not None:
        message = f"<|im_start|>assistant\n{args.reference_text}<|im_end|>\n"
        transcript = tokenizer.encode(message, add_special_tokens=False)[3:-2]
        if not transcript:
            raise ValueError("The reference transcription produced no content tokens")
        reference_frames = len(reference_codes)
    ids = [config["tts_bos_token_id"], config["tts_eos_token_id"], config["tts_pad_token_id"],
           *ids[:3], *transcript, *ids[3:-5], *describe]
    prefix_rows = len(ids) + len(tags) + int(profile == "base") + reference_frames
    if prefix_rows >= 1024:
        raise ValueError("Text and reference exceed the Talker context")
    width = codec[0].shape[1]
    with runner(args, work, "text", len(ids), (len(ids), width)) as project:
        embedded = project(ids)
    bos, eos, pad = embedded[:3]
    tag_states = [codec[0][tags].astype(np.float32)]
    if profile == "base":
        from speaker import log_mels, read_reference

        features = log_mels(read_reference(args.reference_audio))
        with runner(args, work, "speaker", len(features), (1, width)) as encode:
            tag_states.append(encode(features))
    tag_states.append(codec[0][[talker["codec_pad_id"]]].astype(np.float32))
    tag_states = np.concatenate(tag_states)
    tag_states[:-1] += pad
    tag_states[-1] += bos
    end = len(ids) - len(describe)
    text_states = np.concatenate((embedded[6:end], eos[None, :]), axis=0) + codec[0][talker["codec_pad_id"]]
    audio_bos = pad + codec[0][talker["codec_bos_id"]]
    prefix = [embedded[end:], embedded[3:6], tag_states, text_states, audio_bos[None, :]]
    if reference_codes is not None:
        prefix.append(frame_embeddings(codec, reference_codes) + pad)
    prefix = np.concatenate(prefix, axis=0)
    return prefix, pad


def codes(args, work, reference_codes=None):
    config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    vectors = load_file(args.prepared / "codec.safetensors")
    codec = [vectors["codec_embedding.weight"], *(vectors[f"codec_embedding.{i}.weight"] for i in range(15))]
    prefix, pad = prompt(args, work, config, codec, reference_codes)
    width = prefix.shape[1]
    eos = config["talker_config"]["codec_eos_token_id"]
    base = config["tts_model_type"] == "base"
    seed = getattr(args, "seed", None)
    random = np.random.default_rng(seed) if base or seed is not None else None
    sampling = (getattr(args, "temperature", 0.9), getattr(args, "top_k", 50), getattr(args, "top_p", 1.0))
    penalty = getattr(args, "repetition_penalty", 1.05)
    max_frames = min(args.max_frames, 1024 - len(prefix) + 1)
    context = np.zeros((len(prefix) + max_frames - 1, width), dtype=np.float32)
    context[:len(prefix)] = prefix
    frame_context = np.zeros((16, width), dtype=np.float32)
    frames, seen = [], set()
    capacity = 0
    with runner(args, work, "predictor", 16, (15, 2048)) as predict, ExitStack() as talker:
        for step in range(max_frames):
            if step == capacity:
                talker.close()
                capacity = min(max(TALKER_INITIAL_FRAMES, 2 * capacity), max_frames)
                rows = len(prefix) + capacity - 1
                talk = talker.enter_context(runner(args, work, "talker", rows, (rows, width + 3072)))
            last = len(prefix) - 1 + step
            result = talk(context[:rows])[last]
            logits = result[width:].copy()
            for token in seen:
                logits[token] *= penalty if logits[token] < 0 else 1 / penalty
            if random is None:
                first = int(logits[:2048].argmax())
                if logits[eos] > logits[first]:
                    first = eos
            else:
                eos_score = logits[eos]
                logits[2048:] = -np.inf
                if step >= 2:
                    logits[eos] = eos_score
                first = sample_code(logits, random, *sampling)
            if first == eos:
                return np.asarray(frames, dtype=np.int64).reshape(-1, 16), True
            seen.add(first)
            frame = [first]
            frame_context[0], frame_context[1] = result[:width], codec[0][first]
            for group in range(1, 16):
                scores = predict(frame_context)[group - 1]
                code = int(scores.argmax()) if random is None else sample_code(scores, random)
                frame.append(code)
                if group < 15:
                    frame_context[group + 1] = codec[group][code]
            frames.append(frame)
            if step + 1 < max_frames:
                context[last + 1] = frame_embeddings(codec, np.asarray(frame, dtype=np.int64)[None, :])[0] + pad
    return np.asarray(frames, dtype=np.int64), False


def decode_reference(args, codes):
    import torch
    from qwen_tts import Qwen3TTSTokenizer

    decoder = Qwen3TTSTokenizer.from_pretrained(str(args.speech_tokenizer), device_map=args.codec_device, local_files_only=True)
    with torch.inference_mode():
        (audio,), rate = decoder.decode([{"audio_codes": torch.from_numpy(codes)}])
    return audio, rate


def waveform_windows(codes):
    for start in range(0, len(codes), WAVEFORM_CHUNK):
        context = min(start, WAVEFORM_CONTEXT)
        yield codes[start - context:start + WAVEFORM_CHUNK], context * SAMPLES_PER_FRAME


def decode_waveform(args, work, codes):
    if len(codes) == 0:
        raise ValueError("The waveform decoder needs at least one frame")
    audio = []
    for length, windows in groupby(waveform_windows(codes), key=lambda item: len(item[0])):
        with runner(args, work, "waveform", length, (length * SAMPLES_PER_FRAME,)) as infer:
            for window, discard in windows:
                audio.append(infer(window)[discard:])
    return np.concatenate(audio)


def clone_waveform(args, work, frames, reference_codes=None):
    both = frames if reference_codes is None else np.concatenate((reference_codes, frames))
    if args.waveform_weights:
        audio, rate = decode_waveform(args, work, both), WAVEFORM_RATE
    else:
        audio, rate = decode_reference(args, both)
    if reference_codes is not None:
        if rate != WAVEFORM_RATE or len(audio) != len(both) * SAMPLES_PER_FRAME:
            raise ValueError("Unexpected sample count or rate for the ICL waveform")
        audio = audio[len(reference_codes) * SAMPLES_PER_FRAME:]
    return audio, rate


def save_wave(output, audio, rate):
    if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("The speech tokenizer returned an invalid waveform")
    pcm = np.rint(np.clip(audio, -1, 1) * 32767).astype("<i2")
    with output.open("xb") as stream, wave.open(stream, "wb") as wav:
        wav.setparams((1, 2, rate, len(pcm), "NONE", "not compressed"))
        wav.writeframes(pcm.tobytes())
    return {"sample_rate": rate, "samples": len(pcm)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("probe", "model", "prepared", "scratch", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--language", default="auto", type=str.lower)
    parser.add_argument("--speaker", default="", type=str.lower)
    parser.add_argument("--instruction", default="")
    parser.add_argument("--reference-audio", type=Path, help="local recording for Base voice cloning")
    parser.add_argument("--reference-text", default="", help="transcription of the reference recording for ICL")
    parser.add_argument("--audio-encoder-weights", type=Path, help="prepared 12 Hz Mimi audio encoder weights")
    parser.add_argument("--seed", type=int, help="enable reproducible sampling")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--max-frames", default=128, type=int)
    parser.add_argument("--backend", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--expected-device", required=True)
    parser.add_argument("--expected-pjrt")
    waveform = parser.add_mutually_exclusive_group()
    waveform.add_argument("--waveform-weights", type=Path, help="prepared shared ZML 12 Hz waveform weights")
    waveform.add_argument("--speech-tokenizer", type=Path, help="use the official PyTorch speech tokenizer")
    parser.add_argument("--codec-device", default="cpu")
    args = parser.parse_args()
    if not args.text.strip() or any(token in args.text + args.instruction + args.reference_text for token in ("<|im_start|>", "<|im_end|>")):
        parser.error("Enter text without reserved chat markers")
    if not 1 <= args.max_frames <= 1024:
        parser.error("--max-frames must be between 1 and 1024")
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be nonnegative")
    if not (np.isfinite(args.temperature) and args.temperature > 0 and args.top_k >= 0
            and 0 < args.top_p <= 1 and np.isfinite(args.repetition_penalty) and args.repetition_penalty > 0):
        parser.error("Invalid sampling parameters")
    for name in ("probe", "model", "prepared", "scratch"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    if args.scratch.drive.upper() == "C:":
        parser.error("--scratch must be outside C:")
    expected_suffix = ".wav" if args.waveform_weights or args.speech_tokenizer else ".npy"
    if (args.output.suffix.lower() != expected_suffix or args.output.exists() or args.output.is_symlink()
            or not args.output.parent.is_dir()):
        parser.error(f"Choose a new {expected_suffix} path in an existing directory")
    if args.speech_tokenizer:
        args.speech_tokenizer = args.speech_tokenizer.resolve(strict=True)
    if args.waveform_weights:
        args.waveform_weights = args.waveform_weights.resolve(strict=True)
    config = verify_config(args.model, args.profile)
    if config["tts_model_type"] == "base":
        if not args.reference_audio or args.instruction or args.speaker:
            parser.error("Base requires --reference-audio; --speaker and --instruction apply to other profiles")
        verify_speaker_config(args.model, args.profile)
        args.reference_audio = args.reference_audio.resolve(strict=True)
        args.reference_text = args.reference_text.strip()
        if bool(args.reference_text) != bool(args.audio_encoder_weights):
            parser.error("ICL requires both --reference-text and --audio-encoder-weights")
        if args.audio_encoder_weights:
            args.audio_encoder_weights = args.audio_encoder_weights.resolve(strict=True)
    elif args.reference_audio or args.reference_text or args.audio_encoder_weights:
        parser.error("Reference audio and ICL options require a Base checkpoint")
    with redirect_stdout(sys.stderr):
        with TemporaryDirectory(prefix="qwen-voice-", dir=args.scratch) as folder:
            work = Path(folder)
            refs = encode_reference(args, work) if args.reference_text else None
            frames, complete = codes(args, work, refs)
            if not len(frames):
                raise ValueError("The Talker stopped before generating any audio")
            info = {"file": str(args.output.resolve()), "frames": len(frames), "eos": complete, "code_backend": args.backend}
            if config["tts_model_type"] == "base":
                info.update(voice_mode="icl" if refs is not None else "x_vector_only", seed=args.seed)
            if args.waveform_weights or args.speech_tokenizer:
                audio, rate = clone_waveform(args, work, frames, refs)
                info.update(save_wave(args.output, audio, rate), waveform_backend="zml" if args.waveform_weights else "qwen-tts")
            else:
                with args.output.open("xb") as stream:
                    np.save(stream, frames, allow_pickle=False)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
