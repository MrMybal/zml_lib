# Source status

[Français](VERSION.fr.md) · [README](README.md)

## Model ports

- Real-ESRGAN shares RRDBNet across the ×2, ×4 and anime variants, with a C ABI and a JSON CLI.
- The Windows CPU validation covers SRVGGNet, SwinIR-M JPEG10, RealESRGAN_x2plus, Video Depth Anything Small, and two short-context Qwen checkpoints.
- Dolphin 3.0 Llama 3.1 8B matched PyTorch on Windows CPU for two prompts after converting the local GGUF weights to FP16.
- AudioLDM-S Full v2 generates mono audio from prompt tokens and an initial latent. Its CLAP text encoder, DDIM scheduler, UNet, VAE and HiFi-GAN vocoder were compared with the local PyTorch reference.
- SwinIR JPEG10, JPEG20, ColorDN noise25 and SRVGGNet were validated with real checkpoints on the RTX 3090. AudioLDM generated 5,152 samples and Video Depth Anything processed 51 frames in three windows with real weights on the same device. [SwinIR](libraries/swinir/README.md) · [AudioLDM](libraries/audioldm/README.md) · [Video Depth Anything](libraries/video_depth_anything/README.md)
- TripoSR generated a colored PLY with 252 vertices and 488 triangles on the RTX 3090 using the official checkpoint. Triplanes, density and colors were compared with PyTorch. [TripoSR](libraries/triposr/README.md)
- Qwen 3.5 0.8B matched Transformers tokens on Windows CPU. [Qwen 3.5](libraries/qwen35/README.md)
- The T5 XXL and UMT5 XXL encoders matched Transformers FP32 on Windows CPU and RTX 3090 using the FLUX and Wan FP16 checkpoints used by ComfyUI. [T5 and UMT5](libraries/t5/README.md)
- FLUX.1 Schnell uses the ComfyUI FP8 checkpoint, CLIP, T5, four Euler steps and the official VAE. The denoiser and VAE were compared against Black Forest Labs' PyTorch implementation. The direct launcher and ImageStudio CLI produced identical PNGs at 512 × 512 and 1024 × 1024 and at 96 × 64 with the Octree LoRA on the RTX 3090 under Windows. TAEF1 through MCP passed at 96 × 64. [FLUX.1 Schnell](libraries/flux1/README.md)
- Whisper Large v3 Turbo reproduced Transformers text, token IDs and timestamps on Windows CPU and RTX 3090 with the official weights. AudioStudio and PersonaRP use the port for their transcription operations. [Whisper](libraries/whisper/README.md#transcrire-un-fichier-audio)
- Stable Diffusion 1.5 text-to-image uses CLIP-L, the U-Net, deterministic DDIM and the original VAE decoder in FP32. The Windows RTX 3090 run was compared with PyTorch and produced byte-identical 512 × 512 PNGs through the direct launcher and the ImageStudio CLI. [Stable Diffusion 1.5](libraries/stable_diffusion/README.md)
- SD 1.5 Inpainting adds the VAE encoder and the nine-channel U-Net. On the Windows RTX 3090, the 192 × 128 output matched PyTorch within one RGB8 level per channel at strengths 0.75 and 1, with exact preservation outside the mask. At 512 × 512 the direct ZML launcher and the ImageStudio CLI produced byte-identical PNGs. [SD 1.5 Inpainting](libraries/stable_diffusion/README.md#sd-15-inpainting)
- TRELLIS structure decoding matched PyTorch's occupied voxels. The structure and SLAT flows matched PyTorch within `4.77e-7` after four Euler steps on Windows CPU with reduced synthetic weights. [TRELLIS](libraries/trellis/README.md)
- Windows CPU tests with synthetic weights matched PyTorch within `7.16e-7` for DINOv2 conditioning and `2.39e-7` for the TRELLIS mesh, Gaussian and radiance heads. They also read back the three 3D formats. [DINOv2](libraries/dinov2/README.md) · [TRELLIS](libraries/trellis/README.md)
- The complete TRELLIS v1 pipeline ran on Windows `iree_cuda` on the RTX 3090 with synthetic weights and 25 Euler steps per flow. PyTorch matched all 357 occupied voxels and mesh faces, with a maximum tensor difference of `3.55e-6`. [CUDA pipeline replay](libraries/trellis/README.md#chaîne-complète-avec-poids-synthétiques)
- GVHMR recovered SMPL-X meshes for 12 and 312 canonical frames from the tennis clip using the local CyTools checkpoints on Windows. The ZML temporal network ran on the RTX 3090 via `iree_cuda`. The maximum absolute difference in world vertices versus PyTorch was `2.41e-6`. [GVHMR](libraries/gvhmr/README.md)
- GVHMR recovered meshes for 195 moving-camera frames using the local CyTools checkpoints. The pipeline ran from SIFT to SMPL-X on the RTX 3090 via `iree_cuda`. Against the same seeded OpenCV/PyCOLMAP reference, homography refinement reduced the maximum trajectory difference from `6.14°` to `2.90°`. With the same camera signal, the maximum world vertex difference versus PyTorch was `1.43e-5`. [Moving-camera validation](libraries/gvhmr/README.md#raffinement-des-homographies-simplevo)
- The 195-frame GVHMR replay also ran YOLOv8x, ViTPose-H and HMR2 appearance features in CUDA with the local CyTools checkpoints. All three models matched their FP32 references. The maximum world vertex difference was `9.54e-6` against PyTorch with the stabilized angle computation and `5.36e-3` against the upstream pipeline. [Preprocessing validation](libraries/gvhmr/README.md#prétraitement-vidéo-par-zml)
- ViTPose-H now decodes UDP coordinates and scores in the same ZML graph. The 195-frame CUDA replay matched all 3,315 OpenCV scores and had a maximum coordinate difference of `0.000122 px` on the same heatmaps. [UDP validation](libraries/gvhmr/README.md#prétraitement-vidéo-par-zml)
- ViTPose-H and HMR2 share a ZML crop and ImageNet normalization. In the moving-camera CUDA replay through SMPL-X, the 195 FP32 crops matched the OpenCV/PyTorch reference exactly. [Crop validation](libraries/gvhmr/README.md#prétraitement-vidéo-par-zml)
- The shared ViTPose-H and HMR2 graph now also halves the resolution of RGB frames. The 195-frame CUDA replay through SMPL-X matched all 134,784,000 RGB8 values from PyAV. [Resize validation](libraries/gvhmr/README.md#prétraitement-vidéo-par-zml)
- SimpleVO uses the same ZML resize for its keyframes. The standalone 195-frame CUDA test matched PyAV pixels, SIFT descriptors and all 34,100 matches, then verified the SimpleVO trajectory against a SciPy reference. [SimpleVO validation](libraries/gvhmr/README.md#extraction-sift-et-appariement-en-cuda)

## Known limits

- The Qwen 3.5 CUDA build requires the Windows fork's FlashInfer CUTLASS MoE module.
- DINOv2, the TRELLIS v1 structure and SLAT stages, and the three 3D heads passed isolated tests with official weights on the RTX 3090. The full pipeline with official weights remains to be tested.
- The inventory tracks full image and video generation pipelines. Validation of the Linux ZML runtime in PersonaRP Colab remains pending.
- Windows validation is reported per model and backend. The core ABI matrix includes GPU runs.

[Model inventory](docs/model-adapters.json) · [Windows ABI validation](tests/windows/README.md) · [Restoration results](tests/windows/restoration.md) · [Generation results](tests/windows/generation-parts.md)
