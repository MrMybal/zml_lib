# SAM ViT-B

Le port Zig charge le checkpoint ComfyUI `sam_vit_b_01ec64.pth`, avec 177 tenseurs d’encodeur et 137 de décodeur. L’encodeur ViT-B/16 utilise les douze blocs avec attention locale et globale et positions relatives. Il fournit un embedding FP32 `[1,256,64,64]` utilisable pour plusieurs requêtes de points, boîte ou masque de raffinement. [Référence Segment Anything](https://github.com/facebookresearch/segment-anything)

L’encodeur reçoit une image RGB NCHW `[1,3,1024,1024]`. L’image est redimensionnée en conservant le ratio, avec un côté long de 1 024. Les canaux sur l’échelle 0 à 255 sont centrés par `[123.675, 116.28, 103.53]` et divisés par `[58.395, 57.12, 57.375]`, puis l’image est complétée avec des zéros en bas et à droite.

Le décodeur prend un buffer plat contenant l’embedding, un indicateur 0 ou 1 de présence du masque, `256 × 256` valeurs de masque ou des zéros, puis les triplets `(x, y, label)` dans le repère réseau. Les labels valent `0` pour un point négatif, `1` pour un point positif, `2` et `3` pour les coins d’une boîte et `-1` pour le point de remplissage de la référence. La sortie contient quatre plans de logits `256 × 256`, puis quatre scores. Pour revenir à la résolution d’origine, interpoler les logits vers 1 024, recadrer la zone utile, puis interpoler vers la taille source.

## Rejeu Windows x64

    $py = (Join-Path $env:COMFYUI_ROOT '.venv\Scripts\python.exe')
    $original = (Join-Path $env:COMFYUI_ROOT 'models\sams\sam_vit_b_01ec64.pth')
    $reference = (Join-Path $env:COMFYUI_ROOT 'custom_nodes\was-node-suite-comfyui\repos\SAM')
    $image = (Join-Path $env:COMFYUI_ROOT '.venv\Lib\site-packages\matplotlib\mpl-data\sample_data\grace_hopper.jpg')
    $work = 'validation-output\sam-replay'
    $env:ZML_CPU_PJRT_DLL = (Resolve-Path '..\zml_win\bazel-win-tools\pjrt-cpu\pjrt_cpu.dll').Path
    & $py .\libraries\sam\validate.py convert --checkpoint $original --output "$work\weights"
    if ($LASTEXITCODE) { throw 'Conversion SAM en échec' }
    python .\scripts\build.py -- --jobs=8 //libraries/sam:sam_probe --then $py .\libraries\sam\validate.py compare --checkpoint $original --weights "$work\weights" --reference $reference --image $image --probe '{bin}\libraries\sam\sam_probe.exe' --backend cpu --output "$work\cpu"
    if ($LASTEXITCODE) { throw 'Comparaison SAM en échec' }

Sur CPU Windows, le test réutilise le même embedding pour les points puis pour une boîte avec masque de raffinement. L’écart absolu maximal de l’embedding est de `1,23e-6`, celui des logits de `3,66e-4`. L’IoU la plus basse parmi les masques à la résolution d’origine est de `0,9999696`.
