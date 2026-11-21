# Qwen 3 1.7B

Le profil correspond aux SafeTensors de CyTools AudioLibrary. Le décodeur partage le graphe du [port Qwen 2.5](../qwen2/README.md) et ajoute la normalisation Q et K par tête. Les 311 tenseurs incluent la projection de sortie du second fragment.

Le [préparateur](../qwen_decoder/prepare.py) vérifie les deux fragments et peut écrire un fichier FP32. La cible `//libraries/qwen3:qwen3_generate` accepte des identifiants de tokens déjà préparés. Le prototype calcule à nouveau le contexte à chaque token.

Sous Windows x64 sur `cpu:0`, ZML et PyTorch FP32 donnent les mêmes trois tokens sur les deux prompts testés avec les poids CyTools. Les suites obtenues sont `[650, 13916, 40839]` et `[12095, 13, 576]`. [Commande Windows](../../tests/windows/generation-parts.md#qwen-25-et-qwen-3)
