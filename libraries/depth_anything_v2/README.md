# Depth Anything V2 Large

Le paquet `//libraries/depth_anything_v2:model` réutilise [DINOv2](../dinov2/README.md) et lui ajoute la tête DPT. Les quatre cartes de patches sont projetées, mises à l’échelle et fusionnées par des blocs résiduels. La dernière convolution rend une carte de profondeur relative.

`Model.load` lit `depth_anything_v2_vitl_fp16.safetensors`. Le chargeur vérifie les 407 tenseurs, puis les convertit en FP32 pendant leur transfert vers le périphérique. Le checkpoint reste à son emplacement local. Le modèle emprunte la session ZML et réutilise les poids et le graphe entre les appels de même résolution. Libérer le modèle avant la session.

`Model.predict` reçoit un buffer RGB planaire `[1, 3, H, W]` en FP32. Aligner chaque dimension sur un multiple de 14, puis soustraire les moyennes `[0,485, 0,456, 0,406]` et diviser par `[0,229, 0,224, 0,225]` sur des pixels dans `[0, 1]`. L’appelant fournit aussi `H × W` valeurs de sortie FP32. La carte brute est dans le repère de l’image d’entrée. L’application peut la redimensionner et la normaliser pour l’affichage.

## Rejeu Windows x64

Utiliser PowerShell 7 depuis la racine de `zml_lib`. La référence est un checkout local du code PyTorch de Depth Anything V2. L’exemple utilise celui déjà présent dans CyTools.

    $python = (Join-Path $env:COMFYUI_ROOT '.venv\Scripts\python.exe')
    $checkpoint = (Join-Path $env:COMFYUI_ROOT 'models\depthanything\depth_anything_v2_vitl_fp16.safetensors')
    $reference = (Join-Path $env:CYTOOLS_ROOT 'CyTools_VideoToGaussian\runtime\sources\c4g\submodules\cowtracker\cowtracker\thirdparty\DepthAnythingV2')
    $env:ZML_CPU_PJRT_DLL = (Resolve-Path '..\zml_win\bazel-win-tools\pjrt-cpu\pjrt_cpu.dll').Path
    $cpu = @('--@zml//platforms:cuda=false', '--@zml//platforms:rocm=false', '--@zml//platforms:vulkan=false')
    python .\scripts\build.py --target //libraries/depth_anything_v2:depth_probe -- --jobs=3 @cpu --then $python -B .\libraries\depth_anything_v2\validate.py --reference $reference --checkpoint $checkpoint --probe '{bin}\libraries\depth_anything_v2\depth_probe.exe' --backend cpu
    if ($LASTEXITCODE) { throw 'Validation Depth Anything V2 en échec' }

Le script compare les quatre activations et la profondeur aux sorties PyTorch FP32 sur deux images de `28 × 42` et `42 × 28`. Il lance deux inférences sur le même modèle chargé, vérifie le backend et le placement des sorties, puis recharge les poids dans un autre processus et teste `Model.predict`. `--image` accepte un fichier RGB et `--size H W` choisit les dimensions de la comparaison. `--report` écrit un rapport JSON à l’emplacement demandé. Les binaires et les fichiers intermédiaires du test sont supprimés à la fin de l’exécution.

## Résultats Windows x64

La compilation a utilisé Bazel 9.3.0 et le checkout `zml_win` à `ed3be5bffb1e84cdb9daa50b36959de996dfdeff`. Ses sources correspondent au verrou Windows `0bb81d1957fb6438faddd1751566ad73eed97914`. Le test a confirmé le périphérique `cpu:0` pour les six inférences. Les 343 tenseurs DINOv2 et les 64 tenseurs DPT concordent avec le schéma Zig.

| Dimensions H × W | Écart maximal des activations DINOv2 | Écart maximal de la profondeur, rechargement inclus |
| --- | ---: | ---: |
| 28 × 42 | `6,68e-5` | `1,15e-4` |
| 42 × 28 | `7,87e-5` | `5,35e-5` |

Les écarts comparent les sorties FP32 à PyTorch 2.10 avec le même fichier de poids FP16. Les backends GPU et l’exécution FP16 native restent à valider.

Le code de référence est sous Apache-2.0. Le modèle Large est publié sous CC-BY-NC-4.0. [Licence des poids](https://github.com/DepthAnything/Depth-Anything-V2#license) · [Provenance du checkpoint](port.json)
