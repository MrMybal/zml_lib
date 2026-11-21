# Source status

[Français](VERSION.fr.md) · [README](README.md)

## Model ports

- Real-ESRGAN shares RRDBNet across the ×2, ×4 and anime variants, with a C ABI and a JSON CLI.
- The Windows CPU validation covers SRVGGNet, SwinIR-M JPEG10, RealESRGAN_x2plus, Video Depth Anything Small, and two short-context Qwen checkpoints.
- Dolphin 3.0 Llama 3.1 8B matched PyTorch on Windows CPU for two prompts after converting the local GGUF weights to FP16.
- AudioLDM-S Full v2 generates mono audio from prompt tokens and an initial latent. Its CLAP text encoder, DDIM scheduler, UNet, VAE and HiFi-GAN vocoder were compared with the local PyTorch reference.

## Known limits

- The Qwen 3.5 Windows launcher builds. Inference validation awaits local weights.
- SwinIR JPEG20 and ColorDN and the full image and video generation pipelines are listed in the inventory.
- Windows validation is reported per model and backend. The core ABI matrix includes GPU runs. The recent restoration, Qwen and AudioLDM results apply to CPU.

[Model inventory](docs/model-adapters.json) · [Windows ABI validation](tests/windows/README.md) · [Restoration results](tests/windows/restoration.md) · [Generation results](tests/windows/generation-parts.md)
