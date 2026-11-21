"""Compare the local audio vocoder, FLUX preview decoder and CLIP text encoder."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from safetensors.torch import save_file
from torch import nn

from build import ROOT, TEMP_ROOT, scratch_directory

sys.path.insert(0, str(ROOT))
from libraries.audioldm.prepare import save_decoder, save_text


AUDIO = Path("CyTools_AudioStudio/data/models/audioldm/feeb3d14203495a4b6ac0893cbdedb2159b4819c/vocoder")
IMAGE = Path("CyTools_ComfyFlow/data/models/vae_approx/taef1_decoder.pth")


def run_probe(probe, arguments, placements=1):
    result = subprocess.run([str(probe), "cpu", *map(str, arguments)], cwd=ROOT, capture_output=True, text=True, errors="replace")
    print(result.stderr, end="", file=sys.stderr)
    result.check_returncode()
    if result.stderr.count("backend=cpu pjrt=cpu device=cpu:0 shards=1") != placements:
        raise ValueError(f"{probe.name} used an unexpected backend")


def compare_output(output, expected):
    actual = np.fromfile(output, dtype="<f4").reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3, equal_nan=False)
    print(f"{output.stem} cpu:0 max_abs={np.max(np.abs(actual - expected)):.9g}", flush=True)


def compare(probe, arguments, output, expected):
    run_probe(probe, arguments)
    compare_output(output, expected)


def vocoder(directory, work, bins):
    from transformers import SpeechT5HifiGan

    model = SpeechT5HifiGan.from_pretrained(directory, local_files_only=True).float().eval()
    frames = 3
    mel = torch.linspace(-2.0, 0.5, frames * 64).reshape(1, frames, 64)
    with torch.inference_mode():
        expected = model(mel).numpy()
    if expected.shape != (1, 512) or not np.isfinite(expected).all():
        raise ValueError("Expected 512 finite samples at 16 kHz")
    print(f"HiFi-GAN PyTorch {expected.shape}, sample range {expected.min():.7g} to {expected.max():.7g}", flush=True)
    if bins:
        input_path, output = work / "mel.f32le", work / "waveform.f32le"
        mel.numpy().astype("<f4", copy=False).tofile(input_path)
        compare(bins / "libraries/hifigan/hifigan_probe.exe", [directory / "model.safetensors", frames, input_path, output], output, expected)


def decode_audio(directory, work, bins):
    from diffusers import AutoencoderKL
    from transformers import SpeechT5HifiGan

    vae_dir = directory / "vae"
    vocoder_dir = directory / "vocoder"
    vae = AutoencoderKL.from_pretrained(vae_dir, local_files_only=True).float().eval()
    hifigan = SpeechT5HifiGan.from_pretrained(vocoder_dir, local_files_only=True).float().eval()
    latent_frames = 3
    latent = torch.linspace(-1, 1, 8 * latent_frames * 16).reshape(1, 8, latent_frames, 16)
    with torch.inference_mode():
        mel = vae.decode(latent / vae.config.scaling_factor).sample.squeeze(1)
        waveform = hifigan(mel)
    if mel.shape != (1, 12, 64) or waveform.shape != (1, 1952) or not torch.isfinite(waveform).all():
        raise ValueError("Unexpected AudioLDM decoder reference")
    print(f"AudioLDM PyTorch mel={tuple(mel.shape)}, samples={waveform.shape[1]}", flush=True)
    if bins:
        weights = work / "decoder.safetensors"
        save_decoder(directory, weights)
        source = work / "latent.f32le"
        mel_output = work / "mel.f32le"
        waveform_output = work / "waveform.f32le"
        latent.numpy().astype("<f4", copy=False).tofile(source)
        run_probe(
            bins / "libraries/audioldm/audio_decode_probe.exe",
            [weights, vocoder_dir / "model.safetensors", latent_frames, source, mel_output, waveform_output],
            placements=2,
        )
        compare_output(mel_output, mel.numpy())
        compare_output(waveform_output, waveform.numpy())


def clap_text(directory, work, bins):
    from transformers import AutoTokenizer, ClapTextModelWithProjection

    tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer", local_files_only=True)
    model = ClapTextModelWithProjection.from_pretrained(directory / "text_encoder", local_files_only=True).float().eval()
    parameters = {key: value.detach().contiguous() for key, value in model.named_parameters()}
    if len(parameters) != 203:
        raise ValueError(f"Expected 203 CLAP text tensors, found {len(parameters)}")
    tokens = tokenizer("Rain falling on a tin roof.", padding="max_length", max_length=16, return_tensors="pt")
    with torch.inference_mode():
        embedding = torch.nn.functional.normalize(model(**tokens).text_embeds, dim=-1).numpy()
    if embedding.shape != (1, 512) or not np.isfinite(embedding).all():
        raise ValueError("Unexpected CLAP embedding")
    print(f"CLAP PyTorch embedding={embedding.shape}, norm={np.linalg.norm(embedding):.7g}", flush=True)
    if bins:
        weights = work / "clap_text.safetensors"
        output = work / "clap_embedding.f32le"
        save_text(directory, weights)
        compare(bins / "libraries/laion_clap/clap_text_probe.exe", [weights, output, *tokens.input_ids[0].tolist()],
                output, embedding)


def generate_audio(directory, work, bins):
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from transformers import AutoTokenizer, ClapTextModelWithProjection, SpeechT5HifiGan

    tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer", local_files_only=True)
    encoder = ClapTextModelWithProjection.from_pretrained(directory / "text_encoder", local_files_only=True).float().eval()
    denoiser = UNet2DConditionModel.from_pretrained(directory / "unet", local_files_only=True).float().eval()
    vae = AutoencoderKL.from_pretrained(directory / "vae", local_files_only=True).float().eval()
    hifigan = SpeechT5HifiGan.from_pretrained(directory / "vocoder", local_files_only=True).float().eval()
    scheduler = DDIMScheduler.from_pretrained(directory / "scheduler", local_files_only=True)
    steps, guidance, frames = 2, 2.0, 8
    scheduler.set_timesteps(steps)
    initial = torch.randn((1, 8, frames, 16), generator=torch.Generator().manual_seed(2026))
    tokens = tokenizer(["", "Rain falling on a tin roof."], padding="max_length", max_length=16, return_tensors="pt")
    latent = initial
    with torch.inference_mode():
        text = torch.nn.functional.normalize(encoder(**tokens).text_embeds, dim=-1)
        for timestep in scheduler.timesteps:
            predicted = denoiser(latent.repeat(2, 1, 1, 1), timestep, encoder_hidden_states=None, class_labels=text).sample
            negative, positive = predicted.chunk(2)
            latent = scheduler.step(negative + guidance * (positive - negative), timestep, latent).prev_sample
        mel = vae.decode(latent / vae.config.scaling_factor).sample.squeeze(1)
        waveform = hifigan(mel)
    if waveform.shape != (1, 5152) or not torch.isfinite(waveform).all():
        raise ValueError("Invalid AudioLDM generation reference")
    print(f"AudioLDM PyTorch generated {waveform.shape[1]} samples with {steps} DDIM steps", flush=True)
    if bins:
        clap_weights = work / "clap_text.safetensors"
        save_text(directory, clap_weights)
        vae_weights = work / "decoder.safetensors"
        save_decoder(directory, vae_weights)
        source = work / "initial.f32le"
        output_latent = work / "denoised.f32le"
        output_waveform = work / "generated.f32le"
        initial.numpy().astype("<f4", copy=False).tofile(source)
        run_probe(
            bins / "libraries/audioldm/audio_generate_probe.exe",
            [clap_weights, directory / "unet/diffusion_pytorch_model.safetensors", vae_weights,
             directory / "vocoder/model.safetensors", frames, steps, guidance, source, output_latent, output_waveform,
             *tokens.input_ids[1].tolist()],
            placements=5,
        )
        compare_output(output_latent, latent.numpy())
        compare_output(output_waveform, waveform.numpy())


class LatentClamp(nn.Module):
    def forward(self, x):
        return (x / 3).tanh() * 3


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1),
                                  nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1))

    def forward(self, x):
        return (self.conv(x) + x).relu()


def preview(checkpoint, work, bins):
    layers = [LatentClamp(), nn.Conv2d(16, 64, 3, padding=1), nn.ReLU()]
    for _ in range(3):
        layers.extend([Residual(), Residual(), Residual(), nn.Upsample(scale_factor=2, mode="nearest"),
                       nn.Conv2d(64, 64, 3, padding=1, bias=False)])
    layers.extend([Residual(), nn.Conv2d(64, 3, 3, padding=1)])
    model = nn.Sequential(*layers).eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    latent = torch.linspace(-3, 2, 96).reshape(1, 16, 2, 3)
    with torch.inference_mode():
        expected = model(latent).numpy()
    if expected.shape != (1, 3, 16, 24) or not np.isfinite(expected).all():
        raise ValueError("Invalid TAEF1 reference")
    print(f"TAEF1 PyTorch {expected.shape}, RGB range {expected.min():.7g} to {expected.max():.7g}", flush=True)
    if bins:
        weights, input_path, output = work / "weights.safetensors", work / "latent.f32le", work / "rgb.f32le"
        save_file({key: value.detach().contiguous() for key, value in state.items()}, weights)
        latent.numpy().astype("<f4", copy=False).tofile(input_path)
        compare(bins / "libraries/taesd/taef1_probe.exe", [weights, 2, 3, input_path, output], output, expected)


def clip(directory, work, bins):
    from transformers import CLIPTextConfig, CLIPTextModel

    if directory:
        model = CLIPTextModel.from_pretrained(directory, local_files_only=True).float().eval()
    else:
        torch.manual_seed(32)
        model = CLIPTextModel(CLIPTextConfig(hidden_size=768, intermediate_size=3072, num_attention_heads=12)).float().eval()
    config = model.config
    profile = (config.hidden_size, config.intermediate_size, config.num_hidden_layers, config.num_attention_heads,
               config.max_position_embeddings, config.vocab_size, config.hidden_act)
    if profile != (768, 3072, 12, 12, 77, 49408, "quick_gelu"):
        raise ValueError("Expected the CLIP-L text profile with QuickGELU")
    ids = [49406, 31, 261, 49407]
    with torch.inference_mode():
        expected = model(torch.tensor([ids])).last_hidden_state.numpy()
    if not np.isfinite(expected).all():
        raise ValueError("Invalid CLIP reference")
    print(f"CLIP PyTorch {expected.shape}, source={'local' if directory else 'synthetic fixture'}", flush=True)
    if bins:
        weights, output = work / "weights.safetensors", work / "text.f32le"
        save_file({key: value.detach().contiguous() for key, value in model.named_parameters()}, weights)
        compare(bins / "libraries/clip/clip_text_probe.exe", [weights, output, *ids], output, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cytools", type=Path, required=True)
    parser.add_argument("--clip-model", type=Path)
    parser.add_argument("--reference-only", action="store_true")
    default_parts = ("hifigan", "taef1", "clip")
    parser.add_argument("--part", action="append", choices=(*default_parts, "audioldm", "clap_text", "audioldm_generate"))
    args = parser.parse_args()
    bins = None
    if not args.reference_only:
        path = os.environ.get("ZML_BIN_DIR")
        if not path:
            parser.error("Run native validation through scripts/build.py --then")
        bins = Path(path).resolve(strict=True)
        if not bins.is_relative_to(TEMP_ROOT):
            parser.error("Expected compiled binaries under _Temp")
    torch.set_num_threads(4)
    cases = {"hifigan": (vocoder, args.cytools / AUDIO), "taef1": (preview, args.cytools / IMAGE),
             "clip": (clip, args.clip_model), "audioldm": (decode_audio, args.cytools / AUDIO.parent),
             "clap_text": (clap_text, args.cytools / AUDIO.parent),
             "audioldm_generate": (generate_audio, args.cytools / AUDIO.parent)}
    for part in args.part or default_parts:
        validate, path = cases[part]
        with scratch_directory() as work:
            validate(path.resolve(strict=True) if path else None, work, bins)


if __name__ == "__main__":
    main()
