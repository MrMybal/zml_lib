# Décodeur TAEF1

Décodeur de prévisualisation des latents FLUX. Pour un latent FP32 `[1,16,H,W]` déjà réglé selon les paramètres du client, `Decoder.decode` écrit un RGB planaire brut `[1,3,8H,8W]`. Les 67 tenseurs proviennent de `CyTools_ComfyFlow/data/models/vae_approx/taef1_decoder.pth`. [Référence TAESD](https://github.com/madebyollin/taesd) · [Script de comparaison](../../scripts/validate_generation_parts.py)

Sous Windows x64, le latent de test 2 × 3 produit une sortie RGB 16 × 24 avec une erreur absolue maximale de `3.06963921e-6` contre PyTorch FP32 sur `cpu:0`. [Rejeu](../../tests/windows/generation-parts.md)
