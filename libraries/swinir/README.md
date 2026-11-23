# SwinIR-M

Les trois profils partagent six groupes résiduels de six blocs, six têtes et 490 paramètres appris. JPEG10 et JPEG20 utilisent des fenêtres de 7 × 7 et une échelle d’image de 255. ColorDN noise25 utilise des fenêtres de 8 × 8, une échelle de 1 et des tables de biais relatif de forme `[225,6]`. Le graphe recalcule les masques pour la résolution reçue et complète les bords par réflexion. [Référence SwinIR](https://github.com/JingyunLiang/SwinIR/blob/main/main_test_swinir.py)

`Model` charge JPEG10 ou JPEG20 et `ColorDenoise.Model` charge ColorDN. Les deux reçoivent et renvoient des buffers RGB planaires FP32 `[1,3,H,W]`. L’entrée est dans `[0,1]` et chaque dimension est au moins égale à la taille de la fenêtre. La sortie brute est recadrée à la taille source.

Sous Windows x64, chaque profil a été comparé à PyTorch FP32 sur deux tailles, avec deux inférences par taille sur la RTX 3090 via `iree_cuda`. Le poids JPEG10 vient de CyTools et les deux autres viennent de la publication officielle SwinIR. [Checkpoints officiels](https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0) · [Rejeu Windows](../../tests/windows/restoration.md)

| Profil | Dimensions | Écart absolu maximal | SHA-256 du checkpoint |
| --- | --- | ---: | --- |
| JPEG10 | 9×13, 14×21 | `1,32e-6` | `0005b707e0e6f75b4d13c7447e2f184858ddbdcec95aa7eb8c07e8afa1d26bd9` |
| JPEG20 | 9×13, 14×21 | `1,43e-6` | `1b47fa32f358630e3de3430c91296535a0011b16f0518dacd5b847f123023d45` |
| ColorDN noise25 | 9×13, 16×24 | `7,16e-7` | `39e264322ba762682de5acee4705aaeda7077b947204b9ce1899519ebd540724` |

Pour un checkpoint JPEG20 local, fournir `--profile swinir-jpeg20 --checkpoint $checkpoint` à `scripts/validate_restoration.py` et compiler `//libraries/swinir:swinir_probe`. Pour ColorDN, utiliser `--profile swinir-colordn25` avec `//libraries/swinir:swinir_color_dn_probe`. Ajouter `--backend cuda --expected-device 'RTX 3090'` avec le même environnement CUDA que le guide Windows.
