# Validation des modèles de restauration CyTools

Le test utilise les checkpoints locaux de CyTools et compare les sorties FP32 sur CPU à PyTorch et Spandrel. Il compile les cibles dans une seule exécution Bazel. Chaque modèle a son propre dossier temporaire pour les poids convertis, les images et les activations. Ces dossiers sont supprimés après chaque essai, puis le build est supprimé en fin d’exécution.

    $py = (Join-Path $env:COMFYUI_ROOT '.venv\Scripts\python.exe')
    $cytools = $env:CYTOOLS_ROOT
    $env:ZML_CPU_PJRT_DLL = (Resolve-Path '..\zml_win\bazel-win-tools\pjrt-cpu\pjrt_cpu.dll').Path
    $restorationTargets = @(
        '--target', '//libraries/srvggnet:srvggnet_probe',
        '--target', '//libraries/swinir:swinir_probe',
        '--target', '//libraries/realesrgan:realesrgan_cli',
        '--target', '//:zml_lib',
        '--target', '//:zml_lib_cpp'
    )
    $cpu = @('--@zml//platforms:cuda=false', '--@zml//platforms:rocm=false', '--@zml//platforms:vulkan=false')
    python .\scripts\build.py @restorationTargets -- --jobs=8 --host_compilation_mode=fastbuild @cpu --then $py -B .\scripts\validate_restoration.py --cytools $cytools

SRVGGNet et SwinIR sont comparés sur 9 × 13 et 14 × 21. Le test RRDBNet ×2 vérifie les activations, le PNG, l’ABI C et le rechargement sur 9 × 13. L’exemple C++ exécute deux inférences sur 8 × 12.

| Checkpoint CyTools | Erreur absolue maximale de sortie FP32 sur `cpu:0` |
| --- | ---: |
| RealESR-General-x4v3 | `3.21865082e-6` |
| SwinIR-M JPEG10 | `1.31130219e-6` |
| RealESRGAN_x2plus | `1.50502e-6` |
