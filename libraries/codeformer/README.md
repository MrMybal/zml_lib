# Chaîne CodeFormer

Le port Zig implémente l’encodeur VQ, le Transformer à neuf couches, la sélection des 256 indices dans le codebook de 1 024 entrées, le décodeur et les fusions SFT contrôlées par la fidélité `w`. Le profil suit le nœud ComfyUI `facerestore_cf` à 512 × 512 en FP32. Le convertisseur choisit `params_ema` de `facerestore_models/codeformer.pth` et charge 515 tenseurs.

Le client ouvre une session avec les trois bibliothèques. RetinaFace détecte les visages et leurs repères. OpenCV aligne chaque visage sur le gabarit à cinq points. CodeFormer restaure la zone alignée, puis ParseNet fournit le masque sémantique. OpenCV applique la transformation inverse et combine les masques comme le nœud ComfyUI. L’image de sortie garde les dimensions d’origine. [CodeFormer](https://github.com/sczhou/CodeFormer) · [FaceXLib](https://github.com/xinntao/facexlib)

## Build et validation Windows x64

Depuis la racine du dépôt, avec le fork ZML configuré et les poids dans ComfyUI. Le test PyTorch utilise le code installé dans `facerestore_cf`. L’exemple utilise le portrait Astronaut de scikit-image et une image unie qui couvre le cas sans visage.

    $comfy = $env:COMFYUI_ROOT
    $py = Join-Path $comfy '.venv\Scripts\python.exe'
    $models = Join-Path $comfy 'models'
    $node = Join-Path $comfy 'custom_nodes\facerestore_cf'
    $image = Join-Path $comfy '.venv\Lib\site-packages\skimage\data\astronaut.png'
    $out = 'validation-output\faces-replay'
    $env:ZML_CPU_PJRT_DLL = (Resolve-Path '..\zml_win\bazel-win-tools\pjrt-cpu\pjrt_cpu.dll').Path
    & $py .\libraries\codeformer\scripts\checkpoints.py --models $models --output "$out\weights"
    if ($LASTEXITCODE) { throw 'Conversion en échec' }
    python .\scripts\build.py -- --jobs=8 //libraries/codeformer:zml_faces --then $py .\libraries\codeformer\scripts\validate.py --dll '{bin}\libraries\codeformer\zml_faces.dll' --node $node --models $models --weights "$out\weights" --image $image --backend cpu --output "$out\cpu"
    if ($LASTEXITCODE) { throw 'Validation en échec' }

Le test compare les têtes et les cinq repères de RetinaFace, les logits et les indices du codebook de CodeFormer, ainsi que les deux sorties de ParseNet. Il traite plusieurs entrées avec les mêmes poids chargés, compare le rendu complet et vérifie le backend et le placement avant la copie de chaque sortie vers l’hôte. Le dossier de sortie contient `reference.png`, `zml.png` et `report.json`.

## Résultats avec les poids de ComfyUI

La chaîne complète a réussi sur `cpu:0` avec un portrait, deux portraits et une image sans visage. Les 256 indices du codebook correspondent à PyTorch pour chaque visage. L’écart absolu le plus élevé dans les tenseurs comparés est de `4,96e-5` en FP32. L’écart maximal dans l’image finale est de deux niveaux sur 255, avec une erreur absolue moyenne de `0,00037` niveau sur l’image à deux visages.

| Backend forcé | Résultat de la chaîne |
| --- | --- |
| CPU | Validée de bout en bout sur `cpu:0` |
| ROCm/HIP | Sorties brutes de RetinaFace divergentes sur `gfx1036`. Le test de parité échoue. |
| Vulkan | La compilation IREE de RetinaFace échoue sur `arith.select` pendant la génération SPIR‑V. |
| CUDA | IREE bloque au chargement du code avec `CUDA_ERROR_INVALID_CONTEXT` sur la RTX 3090. |

Les journaux Windows locaux et les images de comparaison se trouvent dans `validation-output\comfyui-faces`. Les essais GPU utilisent les variables de backend du [guide Windows](../../tests/windows/README.md).

## Exécution et ABI

    python .\scripts\build.py -- --jobs=8 //libraries/codeformer:zml_faces --then $py .\libraries\codeformer\scripts\restore.py --dll '{bin}\libraries\codeformer\zml_faces.dll' --weights "$out\weights" --backend cpu --input $image --output "$out\restored.png" --fidelity 0.5

Le client utilise NumPy et OpenCV côté hôte. Les trois réseaux sont exécutés par ZML. L’ABI de `zml_faces.dll` permet de charger les trois modèles dans la même session. L’appelant détient les buffers, qui sont empruntés jusqu’au retour de chaque appel synchrone. Une session doit être utilisée par un appel à la fois. La fermeture libère les modèles, les graphes et la session. [En-tête C public](zml_faces.h)
