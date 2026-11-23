# DINOv2 ViT-L/14

Le paquet `//libraries/dinov2:model` fournit l’encodeur du checkpoint Depth Anything V2 Large de ComfyUI. Il définit les noms et les dimensions des 343 tenseurs `pretrained.*`. [Architecture DINOv2](https://github.com/facebookresearch/dinov2) · [Référence Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/depth_anything_v2)

`forward` reçoit les paramètres dans l’ordre de `schema` et une image NCHW FP32 `[1, 3, H, W]` normalisée. `H` et `W` sont des multiples de 14. Le graphe projette les patches et exécute 24 blocs avec 16 têtes. Il interpole les positions de la grille `37 × 37` en bicubique, avec le décalage `0,1` du profil de référence. Les sorties normalisées des blocs `4`, `11`, `17` et `23` ont chacune la forme `[1, 1 + H/14 × W/14, 1024]`. Le premier token est le token de classe.

Pour ce profil sans tokens de registre, les quatre activations ont été comparées à PyTorch avec les poids ComfyUI sur Windows x64 et `cpu:0`. L’écart maximal était de `7,87e-5` en FP32. [Validation et chargement avec Depth Anything V2](../depth_anything_v2/README.md)

## ViT-L/14 avec quatre registres pour TRELLIS v1

`Conditioner(registered)` réutilise les blocs ViT et le patch embedding. Il normalise le RGB dans `[0,1]`, interpole les positions en bicubique avec antialiasing si la grille diffère de `37 × 37`, puis insère les registres entre le token de classe et les patches. Il renvoie la dernière activation `x_prenorm`, normalisée jeton par jeton sur les canaux sans paramètres appris. Pour `518 × 518`, la sortie contient 1 374 jetons de largeur 1 024. [Configuration DINOv2](https://github.com/facebookresearch/dinov2/blob/main/dinov2/hub/backbones.py) · [Conditionnement TRELLIS](https://github.com/microsoft/TRELLIS/blob/main/trellis/pipelines/trellis_image_to_3d.py)

Sous Windows x64, exécuter depuis la racine du dépôt avec un Python disposant de NumPy, PyTorch et SafeTensors. Le test convertit des poids synthétiques F16 en F32 et compare deux images RGB à PyTorch. L’image `42 × 84` réduit un axe de la grille et agrandit l’autre. L’image `70 × 70` conserve la grille du profil de test. Deux inférences par chargement ont été vérifiées sur `cpu:0`. L’écart absolu maximal est de `7,1525574e-7` pour les deux tailles. Ajouter `--checkpoint $env:DINOV2_REG_CHECKPOINT` pour rejouer avec le checkpoint officiel local.

```powershell
$py = (Join-Path $env:COMFYUI_ROOT '.venv\Scripts\python.exe')
$env:ZML_TEMP_ROOT = 'D:\_Project\zml_lib_Temp'
$env:ZML_CPU_PJRT_DLL = (Resolve-Path '..\zml_win\bazel-win-tools\pjrt-cpu\pjrt_cpu.dll').Path
$cpu = @('--@zml//platforms:cuda=false', '--@zml//platforms:rocm=false', '--@zml//platforms:vulkan=false')
& $py -B .\scripts\build.py --target //libraries/dinov2:dinov2_condition -- --jobs=12 --host_compilation_mode=fastbuild @cpu --then $py -B .\libraries\dinov2\validate_condition.py --probe '{bin}\libraries\dinov2\dinov2_condition.exe'
```

Pour produire un fichier de conditionnement à conserver, compiler la même cible sans `--then`. Le convertisseur accepte l’état officiel `.pth` ou `.safetensors` en F16/F32 et vérifie les 344 tenseurs. Le lanceur redimensionne en Lanczos une image RGB préparée par l’appelant puis écrit un `.npy` FP32 `[1374,1024]` accepté par `flow_io.py generate`. Le détourage ou le recadrage de l’objet se fait en amont.

```powershell
$dino = '.\validation-output\bin\libraries\dinov2\dinov2_condition.exe'
$dinoWeights = '.\validation-output\dinov2-vitl14-reg.safetensors'
$env:TRELLIS_CONDITION = '.\validation-output\trellis-condition.npy'
& $py -B .\libraries\dinov2\condition_io.py convert --probe $dino --checkpoint $env:DINOV2_REG_CHECKPOINT --weights $dinoWeights
& $py -B .\libraries\dinov2\condition_io.py encode --probe $dino --weights $dinoWeights --image $env:TRELLIS_IMAGE --output $env:TRELLIS_CONDITION
```

[Provenance et licence DINOv2](https://github.com/facebookresearch/dinov2/blob/main/LICENSE)

Sur la RTX 3090 avec `iree_cuda`, les deux inférences par image ont confirmé le placement GPU avant la copie. Les écarts avec PyTorch FP32 sont de `1,3113022e-6` pour `42 × 84` et de `1,9669533e-6` pour `70 × 70`. [Commandes Windows pour la RTX 3090](../trellis/README.md#rejeu-cuda-sur-rtx-3090)
