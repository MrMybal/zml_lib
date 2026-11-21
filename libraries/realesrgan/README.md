# Real‑ESRGAN

Les variantes ×4 partagent RRDBNet en FP32. Le profil standard utilise 23 blocs et 702 tenseurs. Anime_6B utilise 6 blocs et 192 tenseurs. L’inférence traite l’image entière avec batch 1 et pre-padding 0.

| Clé sans suffixe | Dimensions du poids OIHW |
| --- | --- |
| `conv_first` | 64 × 3 × 3 × 3 |
| `body.i.rdbj.conv1` | 32 × 64 × 3 × 3 |
| `body.i.rdbj.conv2` | 32 × 96 × 3 × 3 |
| `body.i.rdbj.conv3` | 32 × 128 × 3 × 3 |
| `body.i.rdbj.conv4` | 32 × 160 × 3 × 3 |
| `body.i.rdbj.conv5` | 64 × 192 × 3 × 3 |
| `conv_body`, `conv_up1`, `conv_up2`, `conv_hr` | 64 × 64 × 3 × 3 |
| `conv_last` | 3 × 64 × 3 × 3 |

Chaque `.bias` contient autant d’éléments que de canaux de sortie. `i` va de 0 à 22 ou de 0 à 5 et `j` de 1 à 3. Le chargeur vérifie le jeu exact de clés, les dimensions, le type FP32 et les plages de données.

## Poids et référence PyTorch

Placer les checkpoints officiels de [la provenance](models/provenance.json) dans votre dossier de poids. Le convertisseur choisit `params_ema` si présent, sinon `params`, et inscrit la clé et le SHA‑256 du checkpoint dans la sortie. Depuis la racine du dépôt, dans PowerShell 7, avec Python 3.11 et le [build Windows](../../README.md) disponible.

    py -3.11 -m venv .venv
    $python = ".\.venv\Scripts\python.exe"
    & $python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
    & $python -m pip install safetensors "git+https://github.com/XPixelGroup/BasicSR.git@8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a"
    $tools = ".\libraries\realesrgan\scripts"
    $weights = $env:REALESRGAN_WEIGHTS_DIR
    foreach ($model in "RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B") {
        $reference = "validation-output\$model-reference"
        & $python "$tools\convert.py" --model $model --checkpoint "$weights\$model.pth" --output "$weights\$model.safetensors"
        if ($LASTEXITCODE) { throw "Conversion en échec pour $model" }
        & $python "$tools\validate.py" reference --model $model --checkpoint "$weights\$model.pth" --output $reference
        if ($LASTEXITCODE) { throw "Référence PyTorch en échec pour $model" }
    }

`reference` exécute RRDBNet de BasicSR. Il vérifie le `state_dict` et utilise une image non carrée de 13 × 9 ou un PNG RGB8 fourni via `--image`. `compare` contrôle la provenance des poids, l’entrée, six activations, la sortie FP32 et le PNG. Il teste également le cycle de vie du modèle via l’ABI.

Configurer le plugin et ses variables pour le backend testé, selon [le rejeu Windows](../../tests/windows/README.md). Utiliser un nouveau dossier de sortie à chaque vérification. Répéter avec le backend GPU voulu.

    python .\scripts\build.py -- --jobs=8
    if ($LASTEXITCODE) { throw 'Compilation en échec' }
    $backend = "cpu"
    foreach ($model in "RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B") {
        & $python "$tools\validate.py" compare --reference "validation-output\$model-reference" --weights "$weights\$model.safetensors" --cli ".\validation-output\bin\libraries\realesrgan\realesrgan_cli.exe" --dll ".\validation-output\bin\zml_lib.dll" --backend $backend --output "validation-output\$model-$backend"
        if ($LASTEXITCODE) { throw "Comparaison $backend en échec pour $model" }
    }

Les tolérances de départ sont `atol=1e-3`, `rtol=1e-3` et un niveau d’écart maximal sur le PNG. Les checkpoints officiels restent locaux.

## Checkpoints RRDBNet de ComfyUI

Le convertisseur sélectionne `params_ema` pour `RealESRGAN_x4plus` et `4x_RealisticRescaler_100000_G` avec les poids locaux vérifiés. Il remappe les clés séquentielles du checkpoint plat `4x_NMKD-Siax_200k` et contrôle les 702 clés et formes canoniques. Pour les trois fichiers convertis, passer `RealESRGAN_x4plus` au CLI ou `ZML_LIB_REALESRGAN_X4PLUS` à l’ABI.

    $python = (Join-Path $env:COMFYUI_ROOT '.venv\Scripts\python.exe')
    $tools = '.\libraries\realesrgan\scripts'
    $models = (Join-Path $env:COMFYUI_ROOT 'models\upscale_models')
    $env:ZML_CPU_PJRT_DLL = (Resolve-Path '..\zml_win\bazel-win-tools\pjrt-cpu\pjrt_cpu.dll').Path
    $out = 'validation-output\comfyui-rrdb'
    foreach ($name in 'RealESRGAN_x4plus', '4x_RealisticRescaler_100000_G', '4x_NMKD-Siax_200k') {
        & $python "$tools\convert.py" --model $name --checkpoint "$models\$name.pth" --output "$out\$name.safetensors"
        if ($LASTEXITCODE) { throw "Conversion en échec pour $name" }
        & $python "$tools\validate.py" reference --reference-engine spandrel --model $name --checkpoint "$models\$name.pth" --output "$out\$name-reference"
        if ($LASTEXITCODE) { throw "Référence en échec pour $name" }
        & $python "$tools\validate.py" compare --reference "$out\$name-reference" --weights "$out\$name.safetensors" --cli '.\validation-output\bin\libraries\realesrgan\realesrgan_cli.exe' --dll '.\validation-output\bin\zml_lib.dll' --backend cpu --output "$out\$name-cpu"
        if ($LASTEXITCODE) { throw "Comparaison en échec pour $name" }
    }

La comparaison utilise PyTorch via Spandrel sur les checkpoints d’origine et contrôle la sortie FP32, le PNG et le cycle de vie du modèle. Sur Windows x64 avec le backend CPU, l’écart absolu maximal en sortie FP32 est de `5,43e-6` sur les trois checkpoints et les PNG sont identiques. [Provenance des poids](models/provenance.json)

## Profil RealESRGAN ×2

`RealESRGAN_x2plus` réutilise les 23 blocs RRDB du profil ×4. Le pixel-unshuffle ×2 fait passer l’entrée de 3 à 12 canaux avant la première convolution. Les dimensions impaires sont complétées à droite et en bas, puis la sortie est recadrée à `[1,3,2H,2W]`. Sur un axe d’un seul pixel, le pixel de bord est répété.

La variante utilise `ZML_LIB_REALESRGAN_X2PLUS` dans l’ABI C, `RealESRGAN_x2plus` dans la requête JSON et `x2plus` dans l’exemple C++. La sortie du CLI est `realesrgan_x2.png`. Le checkpoint `params_ema` de CyTools VideoToStereo3D a été comparé sur `cpu:0` avec Spandrel. L’écart maximal de la sortie FP32 est de `1,51e-6` et le PNG est identique. Les tests de répétition et de rechargement via l’ABI C ont réussi. [Rejeu Windows](../../tests/windows/restoration.md)

## API C

Voir [l’en-tête public](../../include/zml_lib.h) et [l’exemple C++](../../examples/cpp/main.cpp). Le buffer d’entrée est un tableau contigu de `float` en RGB planaire `[1, 3, H, W]` dans `[0, 1]`. Le buffer de sortie contient les valeurs brutes `[1, 3, scale × H, scale × W]` pour l’échelle du modèle. Les tailles sont comptées en `float`. L’appel est synchrone et l’appelant choisit la durée de vie des buffers. Une nouvelle résolution compile le graphe avec les poids déjà chargés.

## CLI pour CyTools_ImageStudio

Lancer un processus par requête. Envoyer un objet JSON UTF‑8 sur stdin et fermer le flux.

    @{
        model = "RealESRGAN_x4plus"
        backend = "cpu"
        weights_path = (Join-Path $env:REALESRGAN_WEIGHTS_DIR "RealESRGAN_x4plus.safetensors")
        image_path = ".\images\input.png"
        output_dir = ".\images\job-001"
    } | ConvertTo-Json -Compress | & ".\validation-output\bin\libraries\realesrgan\realesrgan_cli.exe"

`backend` accepte `auto` (défaut), `cpu`, `cuda`, `rocm` et `vulkan`. Le CLI décode PNG et JPEG en RGB8 et écrit `realesrgan_x4.png` dans le dossier demandé. Le PNG existant est protégé contre l’écrasement. Les valeurs de sortie sont écrêtées dans `[0, 1]` et converties en 8 bits, avec arrondi à l’entier pair en cas d’égalité.

Le CLI émet un objet JSON suivi d’un saut de ligne sur stdout et les diagnostics sur stderr. Une réussite donne le code 0.

    {"ok":true,"output_path":".\\images\\job-001\\realesrgan_x4.png","width":52,"height":36,"backend":"cpu","pjrt_platform":"cpu"}

Une erreur donne un statut non nul et un objet tel que `{"ok":false,"error":"CheckpointShapeMismatch"}`. `"trace":true` écrit huit fichiers FP32 little-endian avec l’entrée, les six activations et la sortie brute.

## Provenance

[Real‑ESRGAN, BSD‑3‑Clause](third_party/LICENSE-Real-ESRGAN.txt), [BasicSR, Apache‑2.0](third_party/LICENSE-BasicSR.txt), [stb, commit 6b5799b](https://github.com/nothings/stb/tree/6b5799bae8a1fe3792691ef4b72ae26434c3e799) et [licence stb](third_party/LICENSE-stb.txt).
