# Qwen 2.5 0.5B

Le profil correspond à CySearchAssist-CLAP-0.5B dans CyTools AudioLibrary. Le graphe utilise RoPE, attention causale à groupes de têtes, biais QKV, RMSNorm et MLP SwiGLU. Les embeddings d’entrée servent aussi de projection de sortie.

Le [préparateur](../qwen_decoder/prepare.py) vérifie les 290 tenseurs BF16 locaux et peut écrire un fichier FP32 pour le port. `qwen2_generate.exe` accepte `BACKEND WEIGHTS CONTEXT MAX_NEW_TOKEN TOKEN_IDS...`. Il réutilise le graphe compilé à chaque pas et calcule à nouveau le contexte à chaque token.

Sous Windows x64 sur `cpu:0`, ZML et PyTorch FP32 donnent les mêmes trois tokens sur les deux prompts testés avec les poids CyTools. Les suites obtenues sont `[650, 18566, 409]` et `[12095, 13, 12095]`. [Commande Windows](../../tests/windows/generation-parts.md#qwen-25-et-qwen-3)
