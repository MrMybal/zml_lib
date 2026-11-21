# Qwen 3.5

`//libraries/qwen35:qwen35_generate` réutilise les graphes Qwen 3.5 dense et MoE du checkout ZML épinglé. Le profil est choisi d’après `model_type` dans le dossier local SafeTensors. Le lanceur partage la sélection de périphérique, le chargement et le décodage du [port Llama](../llama/README.md).

    $model = $env:QWEN35_MODEL_DIR
    $env:ZML_CPU_PJRT_DLL = (Resolve-Path '..\zml_win\bazel-win-tools\pjrt-cpu\pjrt_cpu.dll').Path
    $cpu = @('--@zml//platforms:cuda=false', '--@zml//platforms:rocm=false', '--@zml//platforms:vulkan=false')
    python .\scripts\build.py --target //libraries/qwen35:qwen35_generate -- --jobs=8 --host_compilation_mode=fastbuild @cpu --then '{bin}\libraries\qwen35\qwen35_generate.exe' cpu $model 128 'Réponds en trois mots.'

Le lanceur compile sous Windows avec un binaire PE AMD64. L’inférence attend un checkpoint Qwen 3.5 local compatible. [Profil Qwen 2.5](../qwen2/README.md) · [Profil Qwen 3](../qwen3/README.md)
