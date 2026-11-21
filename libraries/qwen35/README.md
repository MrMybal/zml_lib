# Qwen 3.5

`//libraries/qwen35:qwen35_generate` réutilise les graphes Qwen 3.5 dense et MoE du checkout ZML épinglé. Le profil est choisi d’après `model_type` dans le dossier local SafeTensors. Le lanceur partage la sélection de périphérique, le chargement et le décodage du [port Llama](../llama/README.md).

    $model = $env:QWEN35_MODEL_DIR
    $env:ZML_CPU_PJRT_DLL = (Resolve-Path '..\zml_win\bazel-win-tools\pjrt-cpu\pjrt_cpu.dll').Path
    $cpu = @('--@zml//platforms:cuda=false', '--@zml//platforms:rocm=false', '--@zml//platforms:vulkan=false')
    python .\scripts\build.py --target //libraries/qwen35:qwen35_generate -- --jobs=8 --host_compilation_mode=fastbuild @cpu --then '{bin}\libraries\qwen35\qwen35_generate.exe' cpu $model 128 'Réponds en trois mots.'

L’analyse Bazel Windows et la compilation ciblée de SentencePiece et SWIG passent. La compilation complète du lanceur et l’inférence avec un checkpoint Qwen 3.5 local restent à valider. [Profil Qwen 2.5](../qwen2/README.md) · [Profil Qwen 3](../qwen3/README.md)
