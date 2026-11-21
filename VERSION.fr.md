# État des sources

[English](VERSION.md) · [README](README.fr.md)

## Portages de modèles

- Real-ESRGAN partage RRDBNet entre les variantes ×2, ×4 et anime, avec une ABI C et un CLI JSON.
- La validation Windows sur CPU couvre SRVGGNet, SwinIR-M JPEG10, RealESRGAN_x2plus, Video Depth Anything Small et deux checkpoints Qwen à contexte court.
- AudioLDM-S Full v2 génère du son mono à partir de tokens et d’un latent initial. Son encodeur CLAP texte, son ordonnanceur DDIM, son UNet, son VAE et son vocodeur HiFi-GAN ont été comparés à la référence PyTorch locale.

## Limites connues

- Llama et Qwen 3.5 attendent une compilation Windows complète et un test d’inférence.
- SwinIR JPEG20 et ColorDN et les chaînes complètes de génération d’images et de vidéos figurent dans l’inventaire.
- Les validations Windows sont détaillées par modèle et backend. La matrice de l’ABI de base comprend des essais GPU. Les résultats récents de restauration, de Qwen et d’AudioLDM concernent le CPU.

[Inventaire des modèles](docs/model-adapters.json) · [Validation de l’ABI Windows](tests/windows/README.md) · [Résultats de restauration](tests/windows/restoration.md) · [Résultats de génération](tests/windows/generation-parts.md)
