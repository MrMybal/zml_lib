# LAION-CLAP

Le port couvre la tour texte de LAION-CLAP utilisée par AudioLDM. Il charge 203 tenseurs FP32 du checkpoint CyTools et calcule un embedding L2 normalisé de dimension 512 à partir des tokens RoBERTa et du masque. Les deux buffers PyTorch de position et de types sont recréés au calcul. L’encodeur audio HTSAT et le tokenizer Zig restent à intégrer.

    $py = (Join-Path $env:COMFYUI_ROOT '.venv\Scripts\python.exe')
    $env:ZML_CPU_PJRT_DLL = (Resolve-Path '..\zml_win\bazel-win-tools\pjrt-cpu\pjrt_cpu.dll').Path
    $cpu = @('--@zml//platforms:cuda=false', '--@zml//platforms:rocm=false', '--@zml//platforms:vulkan=false')
    python .\scripts\build.py --target //libraries/laion_clap:clap_text_probe -- --jobs=8 --host_compilation_mode=fastbuild @cpu --then $py -B .\scripts\validate_generation_parts.py --part clap_text --cytools $env:CYTOOLS_ROOT

Sous Windows x64, l’embedding du prompt de test avec les poids CyTools correspond à PyTorch FP32 sur `cpu:0`. L’erreur absolue maximale est de `1.45286322e-7`.
