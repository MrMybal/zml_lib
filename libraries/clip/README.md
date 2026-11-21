# CLIP texte

Le profil CLIP-L de Stable Diffusion accepte de 1 à 77 identifiants de tokens et produit les activations conditionnelles `[1,L,768]`. Il utilise 12 blocs d’attention causale, QuickGELU et la normalisation finale. Le vocabulaire compte 49 408 entrées. L’appelant fournit les tokens produits par le tokenizer du modèle.

Le [script de comparaison](../../scripts/validate_generation_parts.py) accepte un dossier CLIP texte local via `--clip-model`. En son absence, il génère une référence déterministe avec le même graphe. Le texte complet vers image et l’encodeur vision restent à intégrer. [Référence CLIP de Transformers](https://huggingface.co/docs/transformers/model_doc/clip)

Sous Windows x64, le modèle synthétique de test produit quatre états cachés de largeur 768, avec une erreur absolue maximale de `3.93390656e-6` contre PyTorch FP32 sur `cpu:0`. [Rejeu](../../tests/windows/generation-parts.md)
