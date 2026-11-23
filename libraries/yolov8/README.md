# YOLOv8 pour ComfyUI et GVHMR

La bibliothèque partage le backbone, le cou de détection et le décodage DFL des trois checkpoints locaux. Les profils visage et personne utilisent YOLOv8m, avec deux ou quatre blocs internes par C2f. Le profil main utilise YOLOv8s, avec un ou deux blocs. La variante personne ajoute les 32 coefficients et prototypes de segmentation. Les graphes Zig traitent les convolutions, BatchNorm avec epsilon `1e-3`, SPPF, la fusion des trois échelles et les têtes. [Référence Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/)

L’inférence prend un buffer RGB NCHW `[1,3,H,W]` FP32 normalisé entre 0 et 1 après letterbox. Les dimensions réseau sont des multiples de 32. La sortie est planaire, dans l’ordre centre x, centre y, largeur, hauteur, puis score. Pour personne, suivent les 32 plans de coefficients par ancre et les 32 prototypes de taille `H/4 × W/4`. Le post-traitement de validation effectue la NMS, ramène les boîtes à la résolution d’origine et combine les coefficients et prototypes pour chaque masque.

## Rejeu Windows x64

Depuis la racine du dépôt, utiliser un environnement Python avec Ultralytics 8.3.253, PyTorch, NumPy, OpenCV et SafeTensors. Les fichiers `.pt` Ultralytics sérialisent des objets Python. Le convertisseur charge les checkpoints locaux de confiance et exporte les tenseurs FP32. Les probes Zig lisent les fichiers SafeTensors obtenus.

    $py = (Join-Path $env:CYTOOLS_ROOT 'CyTools_VideoToGaussian\runtime\4danyone-python\Scripts\python.exe')
    $models = (Join-Path $env:COMFYUI_ROOT 'models\ultralytics')
    $assets = (Join-Path $env:CYTOOLS_ROOT 'CyTools_VideoToGaussian\runtime\4danyone-python\Lib\site-packages\ultralytics\assets')
    $work = 'validation-output\yolo-replay'
    $env:YOLO_CONFIG_DIR = (New-Item -ItemType Directory -Force "$work\ultralytics").FullName
    $env:ZML_CPU_PJRT_DLL = (Resolve-Path '..\zml_win\bazel-win-tools\pjrt-cpu\pjrt_cpu.dll').Path
    python .\scripts\build.py -- --jobs=8 //libraries/yolov8:yolov8_probe
    if ($LASTEXITCODE) { throw 'Build YOLOv8 en échec' }
    & $py .\libraries\yolov8\validate.py convert --models $models --output "$work\weights"
    if ($LASTEXITCODE) { throw 'Conversion en échec' }
    foreach ($profile in 'face', 'hand', 'person') {
        $fixture = if ($profile -eq 'hand') { "$assets\zidane.jpg" } else { "$assets\bus.jpg" }
        & $py .\libraries\yolov8\validate.py compare --models $models --weights "$work\weights" --profile $profile --image $fixture --probe '.\validation-output\bin\libraries\yolov8\yolov8_probe.exe' --backend cpu --output "$work\$profile"
        if ($LASTEXITCODE) { throw "Comparaison en échec pour $profile" }
    }

Le test contrôle les tenseurs convertis, le backend réel, les sorties FP32 et les détections après NMS. Il vérifie aussi les masques à la résolution source. Sur CPU Windows, les trois profils produisent respectivement 2 visages, 1 main et 4 personnes sur ces images. L’écart maximal des coordonnées brutes est de `0,00138` pixel réseau. Les quatre masques personne obtiennent une IoU de `1,0` avec la référence.

Le même rejeu avec `--backend cuda` a réussi sur RTX 3090 via `iree_cuda` après l’ajout du profil COCO. Les décomptes et l’IoU sont identiques. L’écart maximal des coordonnées brutes en CUDA est de `0,00540` pixel réseau.

## YOLOv8x COCO pour GVHMR

Le profil `coco` réutilise le graphe des détecteurs ComfyUI, avec trois ou six blocs internes par C2f. Il produit quatre plans de coordonnées et 80 plans de scores, incluant la classe `person` d’indice 0. `yolov8_probe.exe schema coco` affiche les noms et dimensions attendus pour le checkpoint officiel `yolov8x.pt` fourni par CyTools. Le lanceur conserve le modèle en mémoire pour toutes les images du clip.

```text
yolov8_probe.exe cuda coco weights.safetensors width height input.f32 output.f32 frames repeats
yolov8_probe.exe bgr cuda coco weights.safetensors width height input.bgr output.f32 frames repeats
yolov8_probe.exe letterbox cuda width height input.bgr canvas.f32 frames repeats
yolov8_probe.exe letterbox-shape width height
```

Le premier mode reçoit des images `[3,H,W]` en FP32 little-endian après letterbox et normalisation. Le mode `bgr` reçoit des images BGR8 `[H,W,3]` à leur résolution d’origine. Il effectue le redimensionnement bilinéaire entier, le remplissage centré de valeur 114, la permutation BGR vers RGB et la normalisation dans le graphe du détecteur. Le grand côté mesure 640 pixels et le canevas est aligné sur un pas de 32. `letterbox` exécute le même graphe de préparation seul et renvoie les canevas RGB FP32. `letterbox-shape` affiche la largeur et la hauteur du canevas dans cet ordre. [Ultralytics LetterBox](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/augment.py) · [Interpolation OpenCV](https://github.com/opencv/opencv/blob/4.14.0/modules/imgproc/src/resize.cpp)

Les fichiers concatènent les images dans l’ordre de lecture. La sortie brute du détecteur contient `[84,21 × (H/32) × (W/32)]` valeurs FP32 par image, avec `H,W` mesurés sur le canevas du réseau. Le lanceur vidéo utilise l’entrée BGR8 et transmet les détections après NMS ZML à BoT-SORT. [Validation dans GVHMR](../gvhmr/README.md#prétraitement-vidéo-par-zml)

### Validation CUDA du letterbox

`validate_letterbox.py` compare les pixels et les valeurs FP32 à OpenCV. `--video` ajoute le clip local, `--frames` limite le nombre d’images et `--model-smoke` compare la détection à partir des modes BGR8 et FP32 avec les mêmes poids synthétiques. Le test utilise NumPy, OpenCV et SafeTensors. Il supprime les fichiers d’essai à sa sortie.

Sur RTX 3090 sous Windows x64 via `iree_cuda`, les six formats synthétiques et les 195 images de `Tokyo-Walk_rgb.mp4` concordent exactement avec OpenCV 4.12.0. Le clip produit 143 769 600 valeurs RGB FP32 sur des canevas de 640 × 384. Le test à poids synthétiques retrouve exactement les mêmes coordonnées et scores avec les entrées BGR8 et FP32. Le rejeu avec le checkpoint CyTools dépend du runtime local VideoToGaussian. [Mesures](port.json)

```powershell
$py = 'D:\ComfyUI\.venv\Scripts\python.exe'
$video = 'D:\_Project\_CyTools\CyTools_VideoStudio\runtime\engines\stereo\runtime\source\video-depth-anything\assets\example_videos\Tokyo-Walk_rgb.mp4'
$tools = (Resolve-Path '..\zml_win\bazel-win-tools').Path
$env:ZML_TEMP_ROOT = 'D:\_Project\zml_lib_Temp'
$env:CUDA_VISIBLE_DEVICES = '0'
$env:CUDA_CACHE_DISABLE = '1'
$env:ZML_CUDA_PJRT_DLL = Join-Path $tools 'iree-build\python\iree\_pjrt_libs\cuda\iree_pjrt_cuda_dylib.dll'
$env:IREE_PJRT_COMPILER_LIB_PATH = Join-Path $tools 'iree-wheel\iree\compiler\_mlir_libs\IREECompiler.dll'
$env:IREE_PJRT_IREE_COMPILER_OPTIONS = '--iree-cuda-target=sm_86'
$cuda = @('--@zml//platforms:cpu=false', '--@zml//platforms:cuda=true', '--@zml//platforms:rocm=false', '--@zml//platforms:vulkan=false')
& $py -B .\scripts\build.py --target //libraries/yolov8:yolov8_probe `
  -- --jobs=12 --host_compilation_mode=fastbuild @cuda --then $py -B .\libraries\yolov8\validate_letterbox.py `
  --probe '{bin}\libraries\yolov8\yolov8_probe.exe' --video $video --frames 195 --model-smoke --backend cuda --expected-device 'RTX 3090'
```

### NMS de la classe personne en ZML

GVHMR utilise `bgr-nms-source` pour envoyer les prédictions brutes de YOLOv8x à la NMS sur le même périphérique. La NMS conserve les ancres dont la meilleure classe est `person` et dont le score dépasse 0,5. Elle traite les scores par ordre décroissant et départage les égalités par l’ordre des ancres. Elle élimine les recouvrements d’IoU strictement supérieure à 0,7 et conserve au plus 300 boîtes. [Filtrage Ultralytics 8.3.253](https://github.com/ultralytics/ultralytics/blob/v8.3.253/ultralytics/utils/nms.py) · [NMS CPU Torchvision](https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cpu/nms_kernel.cpp)

```text
yolov8_probe.exe bgr-nms cuda coco weights.safetensors width height input.bgr kept.f32 frames repeats [--raw-output raw.f32]
yolov8_probe.exe nms cuda anchors raw.f32 kept.f32 frames repeats
```

`nms` reçoit les prédictions FP32 `[84,A]` de 1 à 8 400 ancres par image. Les deux modes écrivent 300 lignes FP32 little-endian `[x1,y1,x2,y2,score,ancre]` par image. Les lignes de remplissage contiennent cinq zéros et une ancre `-1`. Le lanceur vidéo remplace l’ancre par l’identifiant de classe `0` en transmettant les détections à BoT-SORT. `--raw-output` écrit aussi les prédictions pour comparer la sélection avec la référence.

Le test compare les indices, les coordonnées et les scores à Torchvision CPU. Il couvre les seuils, les égalités de score, les images sans candidat et la limite de 300 boîtes. Le test avec poids synthétiques vérifie le passage du détecteur à la NMS avec et sans copie des prédictions brutes. [Configuration CUDA](#validation-cuda-du-letterbox)

Sous Windows x64 sur RTX 3090 via `iree_cuda`, les 195 jeux de 5 040 prédictions synthétiques ont retenu exactement les mêmes 4 473 boîtes que Torchvision CPU. Le cas de 8 400 ancres a atteint le plafond de 300 boîtes. Les entrées contenant `NaN` ou `Inf` ont renvoyé une erreur native. Sur deux images de `Tokyo-Walk_rgb.mp4` avec des poids synthétiques, le détecteur et la NMS ont reproduit les prédictions brutes et les boîtes de référence. Les résultats sont identiques avec et sans l’option `--raw-output`. [Mesures](port.json)

```powershell
& $py -B .\scripts\build.py --target //libraries/yolov8:yolov8_probe `
  -- --jobs=12 --host_compilation_mode=fastbuild @cuda --then $py -B .\libraries\yolov8\validate_nms.py `
  --probe '{bin}\libraries\yolov8\yolov8_probe.exe' --video $video --model-smoke --backend cuda --expected-device 'RTX 3090'
```

### Coordonnées de la vidéo en ZML

Les modes `bgr-nms-source` et `nms-source` renvoient les boîtes retenues en pixels de la vidéo. Le graphe reproduit le calcul du gain et des marges de `scale_boxes` à partir des dimensions source et réseau, puis borne les coordonnées aux limites de l’image. Les six champs de sortie et le remplissage sont ceux de la NMS. En mode `bgr-nms-source`, `--raw-output` fournit les prédictions dans le repère réseau pour la comparaison de référence. [Reprojection Ultralytics 8.3.253](https://github.com/ultralytics/ultralytics/blob/v8.3.253/ultralytics/utils/ops.py#L94)

```text
yolov8_probe.exe bgr-nms-source cuda coco weights.safetensors width height input.bgr kept.f32 frames repeats [--raw-output raw.f32]
yolov8_probe.exe nms-source cuda anchors width height raw.f32 kept.f32 frames repeats
```

Ajouter `--project` à la commande de validation NMS pour comparer cette reprojection à PyTorch FP32. Les dimensions transmises sont celles de l’image source. Les cas de test couvrent les formats paysage et portrait, les coordonnées au-delà des bords et le recalcul de marge qui peut différer d’un pixel du remplissage d’origine.

Sur RTX 3090 sous Windows x64 via `iree_cuda`, la reprojection a reproduit exactement les coordonnées PyTorch pour 4 473 boîtes sur 195 jeux de prédictions synthétiques et pour 30 boîtes sur cinq autres formats. Deux trames du clip local avec des poids synthétiques ont aussi produit les mêmes coordonnées, scores et indices après détection et reprojection. La comparaison est passée avec et sans `--raw-output`. [Mesures](port.json)
