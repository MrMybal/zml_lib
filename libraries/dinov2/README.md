# DINOv2 ViT-L/14

Le paquet `//libraries/dinov2:model` fournit l’encodeur du checkpoint Depth Anything V2 Large de ComfyUI. Il définit les noms et les dimensions des 343 tenseurs `pretrained.*`. [Architecture DINOv2](https://github.com/facebookresearch/dinov2) · [Référence Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/depth_anything_v2)

`forward` reçoit les paramètres dans l’ordre de `schema` et une image NCHW FP32 `[1, 3, H, W]` normalisée. `H` et `W` sont des multiples de 14. Le graphe projette les patches et exécute 24 blocs avec 16 têtes. Il interpole les positions de la grille `37 × 37` en bicubique, avec le décalage `0,1` du profil de référence. Les sorties normalisées des blocs `4`, `11`, `17` et `23` ont chacune la forme `[1, 1 + H/14 × W/14, 1024]`. Le premier token est le token de classe.

Le profil couvre ViT-L/14 sans tokens de registre. Les quatre activations ont été comparées à PyTorch avec les poids ComfyUI sur Windows x64 et `cpu:0`. L’écart maximal était de `7,87e-5` en FP32. [Validation et chargement avec Depth Anything V2](../depth_anything_v2/README.md)
