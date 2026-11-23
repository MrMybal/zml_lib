# TRELLIS-image-large

Le port TRELLIS-image-large exécute le flot de structure, le décodeur de structure et le flot SLAT en Zig/ZML. Le décodeur de structure `ss_dec_conv3d_16l8_fp16` calcule les convolutions 3D, les blocs résiduels, la normalisation par voxel et les deux agrandissements par pixel shuffle. Le flot SLAT travaille sur les coordonnées occupées qui en résultent. [Configuration de TRELLIS-image-large](https://huggingface.co/microsoft/TRELLIS-image-large/blob/25e0d31ffbebe4b5a97464dd851910efc3002d96/pipeline.json) · [Code de référence](https://github.com/microsoft/TRELLIS)

L’entrée est un latent FP32 contigu `[1,8,D,H,W]`, avec une taille standard de `16×16×16`. La sortie est un champ de logits FP32 `[1,1,4D,4H,4W]`. Les coordonnées `int32` suivent l’ordre `[lot,D,H,W]` dans l’ordre des voxels du tenseur et sont sélectionnées avec `logit > 0`. Le client Zig détient ses buffers et appelle `Decoder.decode` de façon synchrone avant de libérer le modèle et la session.

## Validation Windows x64

Utiliser PowerShell 7 et un Python contenant NumPy, SafeTensors et PyTorch. Le script valide par défaut deux volumes synthétiques rectangulaires et compare les logits et les coordonnées occupées à PyTorch FP32. Il exécute deux inférences par volume chargé.

Sur Windows x64 et `cpu:0`, le latent `3×4×2` a donné 1 119 voxels occupés et un écart maximal des logits de `2,4754554e-6`. Le latent `2×1×3` a donné 275 voxels et un écart de `2,4065375e-6`. Les coordonnées sont identiques à la référence dans les deux cas.

```powershell
$env:ZML_TEMP_ROOT = (Join-Path (Split-Path -Parent (Get-Location).Path) 'zml_lib_Temp')
$env:ZML_CPU_PJRT_DLL = (Resolve-Path '..\zml_win\bazel-win-tools\pjrt-cpu\pjrt_cpu.dll').Path
$py = (Get-Command python).Source
$cpu = @('--@zml//platforms:cuda=false', '--@zml//platforms:rocm=false', '--@zml//platforms:vulkan=false')
& $py -B .\scripts\build.py --target //libraries/trellis:trellis_structure -- --jobs=12 --host_compilation_mode=fastbuild @cpu --then $py -B .\libraries\trellis\validate.py --probe '{bin}\libraries\trellis\trellis_structure.exe'
```

Ajouter `--checkpoint $env:TRELLIS_STRUCTURE_CHECKPOINT --size 16 16 16` à la commande de validation pour utiliser le checkpoint officiel local. La validation promeut les valeurs F16 en F32 dans les deux moteurs. Les noms et dimensions des 74 tenseurs sont comparés à l’en-tête officiel. [Checkpoint 38 tenseurs F32 et 36 F16](https://huggingface.co/microsoft/TRELLIS-image-large/blob/25e0d31ffbebe4b5a97464dd851910efc3002d96/ckpts/ss_dec_conv3d_16l8_fp16.safetensors)

Sur la RTX 3090, le décodeur a chargé les 74 tenseurs officiels et évalué deux fois le latent d’essai de taille `16×16×16`. L’écart absolu maximal des logits face à PyTorch est de `6,11e-4`. Ce latent a donné zéro voxel occupé dans les deux moteurs. L’encodeur DINOv2 ViT-L/14 à quatre registres a également été vérifié sur la RTX 3090 avec ses poids officiels et un tenseur de `518×518`. Les 1 374 jetons présentent un écart maximal de `2,88e-4`. [Validation DINOv2](../dinov2/README.md)

## Produire les coordonnées occupées

Compiler sans `--then` pour disposer du lanceur dans `validation-output\bin`. Fournir un fichier latent NumPy `.npy` et deux nouveaux chemins de sortie.

```powershell
$probe = '.\validation-output\bin\libraries\trellis\trellis_structure.exe'
& $py -B .\libraries\trellis\structure_io.py convert --probe $probe --checkpoint $env:TRELLIS_STRUCTURE_CHECKPOINT --weights '.\validation-output\trellis-structure.safetensors'
& $py -B .\libraries\trellis\structure_io.py decode --probe $probe --weights '.\validation-output\trellis-structure.safetensors' --latent $env:TRELLIS_LATENT --output '.\validation-output\trellis-structure.npz' --backend cpu
```

Le fichier `.npz` contient `logits` et `coordinates`. Les entrées et sorties brutes du lanceur Zig sont des buffers FP32 little-endian contigus. Le script supprime ses fichiers de travail après chaque exécution.

## Diffusion de structure et SLAT

Les deux flots partagent le Transformer à attention croisée et l’échantillonneur d’Euler. Le flot SLAT ajoute les convolutions aux voxels actifs, la réduction vers la grille grossière et la remontée par liaison enfant-parent. Le lanceur charge un modèle par étape, conserve ses poids sur le backend choisi pendant l’échantillonnage et applique le guidage lorsque le temps normalisé est compris entre 0,5 et 1. Le schéma vérifie les 489 tenseurs du flot de structure et les 526 tenseurs du flot SLAT par nom et dimensions.

La génération prend les jetons FP32 de l’[encodeur DINOv2 avec registres](../dinov2/README.md). Pour le profil officiel, le fichier `.npy` contient `[T,1024]`, correspondant à la sortie `x_prenorm` après normalisation sans paramètres. La commande génère le bruit de structure, produit les sites occupés, génère le bruit SLAT sur ces sites puis écrit les coordonnées et les huit caractéristiques SLAT dénormalisées. [Conditionnement et échantillonnage officiels](https://github.com/microsoft/TRELLIS/blob/main/trellis/pipelines/trellis_image_to_3d.py)

Vérifier les deux flots sous Windows avec le même environnement que le décodeur de structure. Ajouter `--structure-checkpoint` ou `--slat-checkpoint` à la commande de test pour utiliser les fichiers officiels locaux. Le test réduit compare quatre pas d’Euler à PyTorch. Sur `cpu:0` sous Windows x64, les écarts maximaux sont de `4,7683716e-7` pour le flot de structure et de `4,2468309e-7` pour le flot SLAT. Sur la RTX 3090, les checkpoints officiels donnent un écart maximal de `5,86e-4` pour la structure et `2,88e-4` pour SLAT après deux pas d’Euler, avec des conditions synthétiques.

```powershell
& $py -B .\scripts\build.py --target //libraries/trellis:trellis_flow -- --jobs=12 --host_compilation_mode=fastbuild @cpu --then $py -B .\libraries\trellis\validate_flows.py --probe '{bin}\libraries\trellis\trellis_flow.exe'
```

Pour générer des latents avec les poids locaux, compiler les cibles `trellis_flow` et `trellis_structure` sans `--then`. Le convertisseur du décodeur de structure figure ci-dessus.

```powershell
$flow = '.\validation-output\bin\libraries\trellis\trellis_flow.exe'
$decoder = '.\validation-output\bin\libraries\trellis\trellis_structure.exe'
$dense = '.\validation-output\trellis-flow-structure.safetensors'
$slat = '.\validation-output\trellis-flow-slat.safetensors'
& $py -B .\libraries\trellis\flow_io.py convert --probe $flow --stage structure --checkpoint $env:TRELLIS_STRUCTURE_FLOW_CHECKPOINT --weights $dense
& $py -B .\libraries\trellis\flow_io.py convert --probe $flow --stage slat --checkpoint $env:TRELLIS_SLAT_FLOW_CHECKPOINT --weights $slat
& $py -B .\libraries\trellis\flow_io.py generate --probe $flow --structure-probe $decoder --structure-flow $dense --structure-decoder '.\validation-output\trellis-structure.safetensors' --slat-flow $slat --condition $env:TRELLIS_CONDITION --output '.\validation-output\trellis-slat.npz'
```

### Représentations finales

La cible `trellis_heads` exécute les trois décodeurs v1 en Zig/ZML. Le tronc commun alterne les fenêtres d’attention creuses de côté 8 et leur décalage de 4. La tête `mesh` subdivise deux fois les voxels, puis produit le champ de distance, les déformations, les poids FlexiCubes et les attributs de couleur. Le code hôte extrait le maillage de taille variable à partir de ce champ creux. Les têtes `gaussian` et `radiance` produisent respectivement 32 gaussiennes et 16 triplets de vecteurs par voxel. Le maillage et le champ radiatif sont exportés en NPZ, les gaussiennes en PLY. [Têtes de référence](https://github.com/microsoft/TRELLIS/tree/main/trellis/models/structured_latent_vae) · [FlexiCubes épinglé](https://github.com/MaxtirError/FlexiCubes/tree/815e075a2a400d06c48d94c347674344ed6ae5c5)

Les en-têtes officiels confirment 120 tenseurs pour `mesh`, 101 pour `gaussian` et 100 pour `radiance`. La conversion locale vérifie le schéma complet. Elle promeut les poids F16 en F32. Les tests utilisent des poids synthétiques, vérifient deux inférences par modèle chargé sur `cpu:0`, comparent les tenseurs à PyTorch, puis relisent les fichiers de sortie. Le script enlève le dossier temporaire de chaque tête. `decode_slat.py` conserve le chemin de comparaison via un checkout TRELLIS v1 PyTorch/CUDA.

| Tête | Écart absolu maximal face à PyTorch | Sortie relue sous Windows x64 |
| --- | ---: | --- |
| Maillage | `1,7136335e-7` | 720 sommets et 1 432 triangles |
| Gaussiennes | `1,7881393e-7` | 288 gaussiennes PLY |
| Champ radiatif | `2,3841858e-7` | 9 voxels Strivec |

Le test d’extraction vérifie aussi une surface fermée, la concordance de ses arêtes et l’orientation des faces. Quatre champs creux supplémentaires ont produit les mêmes faces que FlexiCubes et concordent à `1e-6` près sur les sommets et leurs attributs.

```powershell
$env:ZML_TEMP_ROOT = 'D:\_Project\zml_lib_Temp'
& $py -B .\scripts\build.py --target //libraries/trellis:trellis_heads -- --jobs=12 --host_compilation_mode=fastbuild @cpu --then $py -B .\libraries\trellis\validate_heads.py --probe '{bin}\libraries\trellis\trellis_heads.exe'
```

Avec les checkpoints officiels, ajouter `--kind mesh --checkpoint $checkpoint --backend cuda --expected-device 'RTX 3090'` à `validate_heads.py`. Répéter avec `gaussian` et `radiance` et leurs poids respectifs. Avec neuf cellules SLAT synthétiques, les trois têtes ont été comparées à PyTorch en FP32 sur la RTX 3090, avec deux inférences par tête.

| Tête | Écart absolu maximal face à PyTorch | Sortie relue |
| --- | ---: | --- |
| Maillage | `1,61e-4` | 398 sommets, 780 triangles |
| Gaussiennes | `2,12e-5` | 288 gaussiennes |
| Champ radiatif | `6,03e-4` | 9 cellules Strivec |

Pour décoder un fichier SLAT local, compiler la même cible et utiliser les commandes suivantes pour préparer les poids et exporter la sortie. Répéter avec `--format gaussian` et un `.ply`, ou avec `--format radiance` et un `.npz`.

```powershell
$heads = '.\validation-output\bin\libraries\trellis\trellis_heads.exe'
$meshWeights = '.\validation-output\trellis-mesh.safetensors'
& $py -B .\libraries\trellis\head_io.py convert --probe $heads --format mesh --checkpoint $env:TRELLIS_MESH_CHECKPOINT --weights $meshWeights
& $py -B .\libraries\trellis\head_io.py decode --probe $heads --format mesh --weights $meshWeights --slat '.\validation-output\trellis-slat.npz' --output '.\validation-output\trellis-mesh.npz'
```

Les tables de cas FlexiCubes utilisées par l’extraction sont sous [Apache-2.0](flexicubes/LICENSE.txt). Elles proviennent du sous-module épinglé par TRELLIS.

### Rejeu CUDA sur RTX 3090

Depuis la racine du dépôt, configurer le plugin IREE CUDA et son compilateur livrés avec le checkout local de `zml_win`. Le test demande la plateforme `iree_cuda` et contrôle la résidence de chaque résultat sur la RTX 3090 avant la copie hôte. Les références PyTorch sont calculées en FP32 avec les mêmes poids synthétiques. Chaque commande nettoie ses répertoires de test et de build. [Guide Windows du fork](https://github.com/MrMybal/zml_win/blob/windows-x64/docs/windows-x64.md#cuda-avec-iree-pjrt)

Sur Windows x64, les deux tailles DINOv2 et les trois têtes ont passé deux inférences par chargement sur la RTX 3090 avec le pilote `610.47`, le plugin `iree_cuda` PJRT `0.68` et la cible `sm_86`. Le chargement initial des noyaux échouait avec `CUDA_ERROR_INVALID_CONTEXT`. Le runtime compile désormais les graphes IREE CUDA Windows sur le thread appelant pour conserver le contexte CUDA lors du chargement. [Gestion des contextes CUDA](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__CTX.html)

| Tête | Écart absolu maximal face à PyTorch FP32 | Sortie CUDA relue |
| --- | ---: | --- |
| Maillage | `2,30968e-7` | 720 sommets et 1 432 triangles |
| Gaussiennes | `2,3841858e-7` | 288 gaussiennes PLY |
| Champ radiatif | `2,3841858e-7` | 9 voxels Strivec |

```powershell
$py = (Join-Path $env:COMFYUI_ROOT '.venv\Scripts\python.exe')
$tools = (Resolve-Path '..\zml_win\bazel-win-tools').Path
$env:ZML_TEMP_ROOT = 'D:\_Project\zml_lib_Temp'
$env:CUDA_VISIBLE_DEVICES = '0'
$env:CUDA_CACHE_DISABLE = '1'
$env:ZML_CUDA_PJRT_DLL = (Join-Path $tools 'iree-build\python\iree\_pjrt_libs\cuda\iree_pjrt_cuda_dylib.dll')
$env:IREE_PJRT_COMPILER_LIB_PATH = (Join-Path $tools 'iree-wheel\iree\compiler\_mlir_libs\IREECompiler.dll')
$env:IREE_PJRT_IREE_COMPILER_OPTIONS = '--iree-cuda-target=sm_86'
$cuda = @('--@zml//platforms:cpu=false', '--@zml//platforms:cuda=true', '--@zml//platforms:rocm=false', '--@zml//platforms:vulkan=false')
& $py -B .\scripts\build.py --target //libraries/dinov2:dinov2_condition -- --jobs=12 --host_compilation_mode=fastbuild @cuda --then $py -B .\libraries\dinov2\validate_condition.py --probe '{bin}\libraries\dinov2\dinov2_condition.exe' --backend cuda --expected-device 'RTX 3090'
& $py -B .\scripts\build.py --target //libraries/trellis:trellis_heads -- --jobs=12 --host_compilation_mode=fastbuild @cuda --then $py -B .\libraries\trellis\validate_heads.py --probe '{bin}\libraries\trellis\trellis_heads.exe' --backend cuda --expected-device 'RTX 3090'
```

Ajouter `--kind mesh`, `--kind gaussian` ou `--kind radiance` à la deuxième commande pour isoler une tête.

### Chaîne complète avec poids synthétiques

`validate_pipeline.py` transmet les jetons calculés par DINOv2 au flot de structure. Le décodeur de structure sélectionne les voxels occupés et le flot SLAT les utilise avec les mêmes jetons. L’archive SLAT est relue et alimente les trois têtes. Le test compare cette exécution à une chaîne PyTorch FP32 indépendante avec la même image RGB, les mêmes poids et le même bruit initial. Il exige des coordonnées occupées et des faces identiques, puis relit les deux NPZ et le PLY.

Le profil réduit combine une image synthétique de `42 × 84`, des jetons de largeur 24 et une grille d’occupation de `8 × 8 × 8`. Il applique 25 pas d’Euler par flot et les paramètres de guidage de TRELLIS v1. Chaque évaluation des flots contrôle le périphérique avant de copier le résultat. Le calcul des graphes s’exécute sur CUDA. L’hôte orchestre Euler, la topologie creuse, la triangulation et l’export.

Sous Windows x64, les 93 résultats intermédiaires des graphes résidaient sur la RTX 3090 via `iree_cuda`. La grille comptait les mêmes 357 voxels occupés sur 512 que PyTorch. La chaîne a produit 6 529 sommets et 12 042 triangles, 11 424 gaussiennes PLY et 357 cellules Strivec. Le tableau mesure les écarts absolus sur les tenseurs FP32 avant export.

| Étape | Écart maximal face à PyTorch |
| --- | ---: |
| Conditionnement DINOv2 | `1,3113022e-6` |
| Flot de structure | `4,7683716e-7` |
| Logits d’occupation | `3,5427511e-6` |
| Flot SLAT | `7,1525574e-7` |
| SLAT dénormalisé | `1,9073486e-6` |
| Tête maillage | `4,4703484e-7` |
| Tête gaussiennes | `3,2782555e-7` |
| Tête champ radiatif | `3,5762787e-7` |

Reprendre les variables du [rejeu CUDA](#rejeu-cuda-sur-rtx-3090) et compiler les quatre lanceurs en une commande. Le test et le build suppriment leurs répertoires temporaires, y compris les sorties 3D.

```powershell
& $py -B .\scripts\build.py `
  --target //libraries/dinov2:dinov2_condition `
  --target //libraries/trellis:trellis_flow `
  --target //libraries/trellis:trellis_structure `
  --target //libraries/trellis:trellis_heads `
  -- --jobs=12 --host_compilation_mode=fastbuild @cuda --then $py -B .\libraries\trellis\validate_pipeline.py `
  --bin '{bin}' --backend cuda --expected-device 'RTX 3090'
```
