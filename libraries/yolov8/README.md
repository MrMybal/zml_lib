# YOLOv8 pour ComfyUI

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
