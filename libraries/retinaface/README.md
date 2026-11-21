# RetinaFace ResNet‑50

Le port ZML contient ResNet‑50, FPN, SSH et les trois têtes. Il attend une image planaire BGR en FP32 avec soustraction des moyennes 104, 117 et 123, et produit les boîtes brutes, les probabilités et cinq repères par ancre. Le code Zig décode les ancres aux pas 8, 16 et 32, puis applique le seuil de score 0,8 et la suppression des doublons à 0,4 d’IoU. La hauteur et la largeur peuvent différer. Le modèle garde ses poids en mémoire et recompile le graphe lors d’un changement de résolution.

Le fichier ComfyUI `facedetection/detection_Resnet50_Final.pth` fournit 383 tenseurs FP32 d’inférence après suppression du préfixe `module.` et des 73 compteurs BatchNorm. Les moyennes et variances BatchNorm sont conservées.

La bibliothèque expose les têtes brutes et les détections décodées via le module Zig et l’ABI de la chaîne CodeFormer. [Commande de conversion et validation](../codeformer/README.md) · [Référence FaceXLib](https://github.com/xinntao/facexlib)
