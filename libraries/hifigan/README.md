# Vocodeur HiFi-GAN

Le vocodeur du checkpoint AudioLDM de CyTools convertit un log-mel FP32 `[1,T,64]` en audio mono à 16 kHz. Il charge les 196 tenseurs du SafeTensors public du vocodeur. Les cinq étages de suréchantillonnage donnent `160 × T + 32` échantillons pour ce profil. `Model.synthesize` écrit dans le buffer de l’appelant.

Le graphe couvre le vocodeur. Le modèle de diffusion et le VAE d’AudioLDM restent à porter. [Référence SpeechT5 HiFi-GAN](https://huggingface.co/docs/transformers/model_doc/speecht5#transformers.SpeechT5HifiGan) · [Script de comparaison](../../scripts/validate_generation_parts.py)

Sous Windows x64, le test à trois trames log-mel produit 512 échantillons avec une erreur absolue maximale de `3.63588333e-6` contre PyTorch FP32 sur `cpu:0`. [Rejeu](../../tests/windows/generation-parts.md)
