# TripoSR

TripoSR génère un maillage couleur à partir d’une image. ZML exécute l’encodeur DINO ViT-B/16 inclus dans le checkpoint, le Transformer à 16 blocs, l’agrandissement des trois plans et le MLP qui produit la densité et le RGB. Zig effectue l’échantillonnage bilinéaire des plans côté hôte. Python prépare l’image, extrait l’isosurface avec `scikit-image` puis écrit le PLY. Toute image RGBA est composée sur un fond gris. Le détourage automatique et la texture UV relèvent de l’application cliente. [TripoSR](https://github.com/VAST-AI-Research/TripoSR) · [Configuration officielle](https://huggingface.co/stabilityai/TripoSR/blob/5b52193/config.yaml)

## Compiler et vérifier sous Windows x64

Depuis la racine du dépôt, utiliser un Python avec `numpy`, `Pillow`, `scikit-image`, `torch` et `safetensors`. Le test utilise le plugin PJRT CPU du fork Windows.

```powershell
$env:ZML_TEMP_ROOT = (Join-Path (Split-Path -Parent (Get-Location).Path) 'zml_lib_Temp')
$env:ZML_CPU_PJRT_DLL = (Resolve-Path '..\zml_win\bazel-win-tools\pjrt-cpu\pjrt_cpu.dll').Path
$py = (Get-Command python).Source
$cpu = @('--@zml//platforms:cuda=false', '--@zml//platforms:rocm=false', '--@zml//platforms:vulkan=false')
& $py -B .\scripts\build.py --target //libraries/triposr:triposr_probe -- --jobs=8 --host_compilation_mode=fastbuild @cpu --then $py -B .\libraries\triposr\validate.py --probe '{bin}\libraries\triposr\triposr_probe.exe'
```

La validation compare les triplans, la densité et les couleurs à PyTorch avec les mêmes poids FP32. Elle effectue deux encodages dans la même session, traverse plusieurs lots de requêtes et vérifie le PLY généré. Les poids synthétiques, le maillage de test et les fichiers de compilation sont supprimés à la fin.

Sous Windows x64 sur `cpu:0`, le profil synthétique donne un écart absolu maximal de `5,364418e-7` sur les triplans et de `5,9604645e-8` sur le champ de densité et de couleurs avant quantification. Le PLY contient 4 155 sommets et 7 752 triangles. Les positions ont été vérifiées sur l’isosurface et les couleurs à la précision d’un canal 8 bits.

Pour remplacer le profil synthétique par le checkpoint officiel, ajouter `--checkpoint $env:TRIPOSR_CHECKPOINT --image $env:TRIPOSR_IMAGE` à la commande de validation. Fournir une image préparée et un checkpoint `.ckpt` officiel sur le disque, avec suffisamment de mémoire pour la référence PyTorch et les tenseurs ZML FP32. L’option `--threshold` ajuste l’isosurface. L’en-tête officiel contient 549 tenseurs FP32. Les 547 utilisés sont validés par nom et dimensions. Les deux tenseurs du pooler DINO servent à une sortie auxiliaire. [Checkpoint officiel et licence du modèle](https://huggingface.co/stabilityai/TripoSR/tree/5b52193)

Avec le checkpoint officiel et `examples/chair.png`, la RTX 3090 via `iree_cuda` a produit un PLY de 252 sommets et 488 triangles. L’écart absolu maximal des triplans est de `9,03e-3` sur des activations dont l’amplitude atteint plusieurs milliers. La comparaison des triplans applique une tolérance absolue minimale proportionnelle à leur amplitude globale. La densité, les valeurs à l’isosurface et les couleurs RGB8 sont vérifiées dans leurs unités de sortie. L’écart maximal des couleurs aux sommets est de `1,96e-3` sur `[0,1]`.

## Reconstruire un maillage

Exécuter la commande de build sans `--then` ni la commande qui le suit pour copier le lanceur dans `validation-output\bin`. `--weights` désigne un nouveau dossier lors de la conversion, puis ce même dossier pour l’inférence.

```powershell
$probe = '.\validation-output\bin\libraries\triposr\triposr_probe.exe'
& $py -B .\libraries\triposr\mesh.py convert --probe $probe --checkpoint $env:TRIPOSR_CHECKPOINT --weights '.\validation-output\triposr-weights'
& $py -B .\libraries\triposr\mesh.py mesh --probe $probe --backend cpu --weights '.\validation-output\triposr-weights' --input $env:TRIPOSR_IMAGE --output '.\validation-output\triposr.ply' --resolution 128
```

En cas de réussite, stdout contient un seul objet JSON avec le chemin du PLY et le nombre de sommets et de triangles. Les diagnostics sont sur stderr et un échec renvoie un code de sortie non nul. Le PLY couleur est en binaire little-endian, avec coordonnées XYZ dans le repère du modèle, couleurs RGB `uint8` et indices de sommets à base zéro.

Le module Zig expose un encodeur et un décodeur de champ. L’encodeur reçoit du RGB normalisé en `[1,3,512,512]` et écrit les trois plans XY, XZ et YZ en `[3,40,64,64]`. Le décodeur prend les plans et un buffer de rangées XYZ. Il renvoie une rangée FP32 par point dans l’ordre densité, R, G, B. Les buffers hôte appartiennent à l’appelant. Les appels synchrones empruntent ces buffers jusqu’au retour et vérifient le placement du résultat avant sa copie vers l’hôte. Libérer le modèle avant de fermer la session ZML.
