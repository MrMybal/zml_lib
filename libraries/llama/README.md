# Llama

`//libraries/llama:llama_generate` réutilise le graphe Llama et le cache KV du checkout ZML épinglé. Le programme utilise un dossier local contenant `config.json`, `tokenizer.json` et les poids SafeTensors. Il accepte un prompt et un budget de contexte et écrit le texte sur stdout.

    $model = $env:LLAMA_MODEL_DIR
    $env:ZML_CPU_PJRT_DLL = (Resolve-Path '..\zml_win\bazel-win-tools\pjrt-cpu\pjrt_cpu.dll').Path
    $cpu = @('--@zml//platforms:cuda=false', '--@zml//platforms:rocm=false', '--@zml//platforms:vulkan=false')
    python .\scripts\build.py --target //libraries/llama:llama_generate -- --jobs=8 --host_compilation_mode=fastbuild @cpu --then '{bin}\libraries\llama\llama_generate.exe' cpu $model 128 'Réponds en trois mots.'

Le checkpoint Dolphin 3.0 Llama 3.1 8B de CyTools est en GGUF Q4_K_M. Le [préparateur GGUF](prepare_gguf.py) le convertit en SafeTensors FP16 pour l’inférence. Les 292 tenseurs ont été convertis et le lanceur Windows x64 a chargé 14,96 Gio de poids sur `cpu:0`. [Exemple Llama du moteur ZML](https://github.com/MrMybal/zml_win/tree/windows-x64/examples/llm)

Le [test de génération](../../scripts/validate_llama.py) valide l’entrée texte sur un petit modèle déterministe et compare les trois tokens d’un second modèle pour vérifier le cache KV.

    $py = (Join-Path $env:COMFYUI_ROOT '.venv\Scripts\python.exe')
    python .\scripts\build.py --target //libraries/llama:llama_generate -- --jobs=8 --host_compilation_mode=fastbuild @cpu --then $py -B .\scripts\validate_llama.py --probe '{bin}\libraries\llama\llama_generate.exe'

La validation du GGUF local prépare les poids denses dans un dossier d’essai, génère la référence avec PyTorch, puis compare les identifiants des tokens à ceux du lanceur sur deux prompts. Définissez `LLAMA_GGUF` sur le chemin du checkpoint. Le processus supprime les poids convertis en fin d’essai.

Sur Windows x64, ZML et PyTorch ont produit les mêmes suites `[12366, 11, 323]` et `[220, 20, 11]` sur ces deux prompts.

    $env:ZML_TEMP_ROOT = Join-Path (Split-Path -Parent (Get-Location).Path) 'zml_lib_Temp'
    python .\scripts\build.py --target //libraries/llama:llama_generate -- --jobs=8 --host_compilation_mode=fastbuild @cpu --then $py -B .\scripts\validate_llm_windows.py --llama-gguf $env:LLAMA_GGUF

Le mode `--token-ids ID...` accepte une séquence déjà encodée et renvoie les tokens générés sous forme de tableau JSON. Les clients peuvent ainsi appliquer le tokenizer et le format de conversation propres à leur checkpoint.
