# SwinIR-M JPEG10

Le profil CyTools restaure un RGB comprimé en JPEG avec six groupes résiduels de six blocs. Chaque fenêtre 7 × 7 utilise six têtes et un biais de position relatif. Le graphe décale une fenêtre sur deux, recalcule les masques pour la résolution reçue et complète les bords par réflexion. Le port charge les 490 paramètres appris du checkpoint. Les indices et masques sont calculés dans le graphe. [Référence SwinIR](https://github.com/JingyunLiang/SwinIR)

`Model.infer` reçoit et renvoie des buffers RGB planaires FP32 `[1,3,H,W]`. L’entrée est dans `[0,1]` et chaque dimension doit être d’au moins sept pixels. La sortie brute est recadrée à la taille source.

La référence PyTorch a chargé `006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg10.pth` depuis CyTools VideoStudio, SHA-256 `0005b707e0e6f75b4d13c7447e2f184858ddbdcec95aa7eb8c07e8afa1d26bd9`. La comparaison Windows x64 sur `cpu:0` donne un écart absolu maximal de `1,32e-6` pour les deux tailles testées. JPEG20 et le débruitage restent à valider avec leurs checkpoints. [Rejeu Windows](../../tests/windows/restoration.md)
