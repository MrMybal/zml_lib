# RealPLKSR avec DySample

Le premier profil porte `4xNomos2_realplksr_dysample` en FP32. Le paquet charge 346 tenseurs, exécute 28 blocs DCCM avec Mish, convolution partielle 17 × 17, attention élément par élément et GroupNorm. DySample applique les offsets appris à quatre groupes de canaux et effectue l’interpolation bilinéaire avec prolongement aux bords.

`Model.load` emprunte une session ZML et charge les poids pour une résolution donnée. `Model.infer` reçoit des tranches de l’appelant et vérifie le placement avant la copie vers l’hôte. L’entrée est RGB planaire `[1, 3, H, W]` en `[0, 1]`. La sortie FP32 brute est `[1, 3, 4H, 4W]`. Le modèle réutilise ses poids et son exécutable entre les appels. Libérer le modèle avant la session.

## Build et comparaison PyTorch

La comparaison a utilisé PyTorch 2.10.0, Spandrel 0.4.1 et SafeTensors dans l’environnement de ComfyUI. Exécuter ces commandes depuis la racine de `zml_lib` dans PowerShell avec un nouveau dossier de résultats.

    $env:ZML_CPU_PJRT_DLL = (Resolve-Path '..\zml_win\bazel-win-tools\pjrt-cpu\pjrt_cpu.dll').Path
    $python = (Join-Path $env:COMFYUI_ROOT '.venv\Scripts\python.exe')
    $checkpoint = (Join-Path $env:COMFYUI_ROOT 'models\upscale_models\4xNomos2_realplksr_dysample.pth')
    $reference = 'validation-output\realplksr-13x9'
    & $python .\libraries\realplksr\scripts\validate.py reference --checkpoint $checkpoint --output $reference
    if ($LASTEXITCODE) { throw 'Référence PyTorch en échec' }
    python .\scripts\build.py -- --jobs=8 //libraries/realplksr:realplksr_probe --then $python .\libraries\realplksr\scripts\validate.py compare --reference $reference --probe '{bin}\libraries\realplksr\realplksr_probe.exe' --backend cpu --output "$reference\actual-cpu.f32le"
    if ($LASTEXITCODE) { throw 'Comparaison en échec' }

La commande `reference` convertit le checkpoint local et prépare une image structurée, variée selon les deux axes. `compare` vérifie le backend forcé, l’empreinte de l’entrée, les métadonnées des poids et la sortie. Le probe réutilise le modèle chargé pour une seconde inférence et libère les buffers et le modèle en quittant.

## Résultats Windows x64

| Backend forcé | Plateforme PJRT | Écart absolu maximal FP32 sur 13 × 9 et 1 × 1 |
| --- | --- | --- |
| CPU | `cpu` | `1,14e-6` |
| ROCm/HIP | `iree_rocm` | `1,41e-6` |
| Vulkan | `iree_vulkan` | `1,61e-6` |
| CUDA | `iree_cuda` | Échec `CUDA_ERROR_INVALID_CONTEXT` durant le chargement du code sur la RTX 3090 |

Le graphe regroupe les indices de DySample dans un seul axe pour produire un `gather` compatible avec la version de StableHLO des plugins IREE Windows. Les journaux locaux sont dans `validation-output\comfyui-realplksr\*-portable-*.log`. Configurer les plugins GPU selon le [guide de rejeu Windows](../../tests/windows/README.md).

Le profil de poids est publié par Philip Hofmann sous le tag de licence CC BY 4.0. La référence PyTorch utilise l’architecture RealPLKSR de Spandrel. [Fiche de 4xNomos2](https://huggingface.co/Phips/4xNomos2_realplksr_dysample) · [Source RealPLKSR dans Spandrel](https://github.com/chaiNNer-org/spandrel/blob/main/libs/spandrel/spandrel/architectures/PLKSR/__arch/RealPLKSR.py)
