# Qwen 3.5

`//libraries/qwen35:qwen35_generate` réutilise les graphes Qwen 3.5 dense et MoE du checkout ZML épinglé. Le profil est choisi d’après `model_type` dans le dossier local SafeTensors. Le lanceur partage la sélection de périphérique, le chargement et le décodage du [port Llama](../llama/README.md).

    $model = $env:QWEN35_MODEL_DIR
    $env:ZML_CPU_PJRT_DLL = (Resolve-Path '..\zml_win\bazel-win-tools\pjrt-cpu\pjrt_cpu.dll').Path
    $cpu = @('--@zml//platforms:cuda=false', '--@zml//platforms:rocm=false', '--@zml//platforms:vulkan=false')
    python .\scripts\build.py --target //libraries/qwen35:qwen35_generate -- --jobs=8 --host_compilation_mode=fastbuild @cpu --then '{bin}\libraries\qwen35\qwen35_generate.exe' cpu $model 128 'Réponds en trois mots.'

Le lanceur a été validé sous Windows x64 avec le checkpoint dense Qwen3.5-0.8B officiel et `cpu:0`. Avec les mêmes poids BF16, Transformers 5.3.0 et ZML ont choisi les mêmes trois tokens pour chacun des deux prompts. Les suites obtenues sont `[11751, 13, 198]` et `[220, 20, 11]`. [Checkpoint épinglé](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc0636)

Pour reproduire la comparaison, définir `ZML_TEMP_ROOT` sur `D:\_Project\zml_lib_Temp` et remplacer la commande située après `--then` par celle-ci.

    $py -B .\libraries\qwen35\validate.py --probe '{bin}\libraries\qwen35\qwen35_generate.exe' --model $model --backend cpu --expected-device 'cpu:0'

La compilation CUDA du lanceur combiné dense et MoE échoue sur le fork Windows épinglé lors de la résolution du module `platforms/cuda/flashinfer_cutlass_moe`. L’inférence MoE et la validation CUDA restent ouvertes. [Profil Qwen 2.5](../qwen2/README.md) · [Profil Qwen 3](../qwen3/README.md)
