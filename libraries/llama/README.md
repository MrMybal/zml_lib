# Llama

`//libraries/llama:llama_generate` réutilise le graphe Llama et le cache KV du checkout ZML épinglé. Le programme utilise un dossier local contenant `config.json`, `tokenizer.json` et les poids SafeTensors. Il accepte un prompt et un budget de contexte et écrit le texte sur stdout.

    $model = $env:LLAMA_MODEL_DIR
    $env:ZML_CPU_PJRT_DLL = (Resolve-Path '..\zml_win\bazel-win-tools\pjrt-cpu\pjrt_cpu.dll').Path
    $cpu = @('--@zml//platforms:cuda=false', '--@zml//platforms:rocm=false', '--@zml//platforms:vulkan=false')
    python .\scripts\build.py --target //libraries/llama:llama_generate -- --jobs=8 --host_compilation_mode=fastbuild @cpu --then '{bin}\libraries\llama\llama_generate.exe' cpu $model 128 'Réponds en trois mots.'

Le poids Dolphin 3 local de CyTools est en GGUF Q4_K_M. Le [préparateur GGUF](prepare_gguf.py) a reconnu sa configuration à 32 couches et son tokenizer. L’option `--output` permet de déquantifier les poids vers un dossier SafeTensors FP16 local. L’export complet et l’inférence Windows avec Dolphin 3 restent à vérifier. [Exemple Llama du moteur ZML](https://github.com/MrMybal/zml_win/tree/windows-x64/examples/llm)

Le [test de génération](../../scripts/validate_llama.py) construit un petit modèle SafeTensors déterministe dans `_Temp`. Le modèle PyTorch et le tokenizer BPE de test ont été vérifiés. L’analyse Bazel Windows et la compilation ciblée de SentencePiece et SWIG passent. Le test d’inférence ZML attend la compilation complète du lanceur.

    $py = (Join-Path $env:COMFYUI_ROOT '.venv\Scripts\python.exe')
    python .\scripts\build.py --target //libraries/llama:llama_generate -- --jobs=8 --host_compilation_mode=fastbuild @cpu --then $py -B .\scripts\validate_llama.py --probe '{bin}\libraries\llama\llama_generate.exe'
