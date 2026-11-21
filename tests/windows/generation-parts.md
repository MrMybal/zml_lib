# Composants de génération

Les tests comparent séparément l’encodeur texte CLIP-L, le décodeur d’images TAEF1 et le vocodeur HiFi-GAN. TAEF1 et HiFi-GAN utilisent les checkpoints locaux de CyTools. Pour CLIP-L, passer `--clip-model` avec un dossier de poids local ou utiliser le modèle déterministe du script.

    $py = (Join-Path $env:COMFYUI_ROOT '.venv\Scripts\python.exe')
    $env:ZML_CPU_PJRT_DLL = (Resolve-Path '..\zml_win\bazel-win-tools\pjrt-cpu\pjrt_cpu.dll').Path
    $targets = @('--target', '//libraries/clip:clip_text_probe',
                 '--target', '//libraries/taesd:taef1_probe',
                 '--target', '//libraries/hifigan:hifigan_probe')
    $cpu = @('--@zml//platforms:cuda=false', '--@zml//platforms:rocm=false', '--@zml//platforms:vulkan=false')
    python .\scripts\build.py @targets -- --jobs=8 --host_compilation_mode=fastbuild @cpu --then $py -B .\scripts\validate_generation_parts.py --cytools $env:CYTOOLS_ROOT

Les essais suppriment leurs entrées et sorties intermédiaires sous `_Temp`. [Rejeu Video Depth Anything](../../libraries/video_depth_anything/README.md) · [Port Llama](../../libraries/llama/README.md)

| Port | Entrée testée | Erreur absolue maximale FP32 sous Windows sur `cpu:0` |
| --- | --- | ---: |
| CLIP texte | Modèle synthétique, 4 tokens | `3.93390656e-6` |
| TAEF1 | Poids CyTools, latent 2 × 3 | `3.06963921e-6` |
| HiFi-GAN | Poids CyTools, 3 trames log-mel | `3.63588333e-6` |
| Video Depth Anything Small | Poids CyTools, clips de 2 et 3 images | `5.96046448e-6` |
| AudioLDM VAE → HiFi-GAN | Poids CyTools, latent de 3 trames | `4.76837158e-6` log-mel, `2.53785402e-8` signal |
| LAION-CLAP texte | Poids CyTools, prompt de 16 tokens avec remplissage | `1.45286322e-7` |
| AudioLDM-S Full v2 | Poids CyTools, prompt et bruit initial, 2 pas DDIM avec guidage | `1.23977661e-5` latent, `2.15694308e-6` signal |

Le test AudioLDM compare la sortie du VAE et le signal après passage dans le vocodeur. Le test complet compare aussi le latent débruité sur deux pas DDIM avec guidage. [Rejeu des deux tests](../../libraries/audioldm/README.md)

## Qwen 2.5 et Qwen 3

Le script charge le tokenizer local et compare trois tokens gloutons en FP32 sur deux prompts. Chaque essai crée un fichier temporaire de poids FP32, environ 2 Go pour Qwen 2.5 et 8,1 Go pour Qwen 3, puis le supprime.

    $models = (Join-Path $env:CYTOOLS_ROOT 'CyTools_AudioLibrary\data\models')
    $qwen2 = "$models\cymetasound-search-assist-0.5b\v1\_dequantized"
    $qwen3 = "$models\qwen3-1.7b\2025-04"
    python .\scripts\build.py --target //libraries/qwen2:qwen2_generate -- --jobs=8 --host_compilation_mode=fastbuild @cpu --then $py -B .\scripts\validate_qwen.py --profile qwen2 --model $qwen2 --probe '{bin}\libraries\qwen2\qwen2_generate.exe'
    python .\scripts\build.py --target //libraries/qwen3:qwen3_generate -- --jobs=8 --host_compilation_mode=fastbuild @cpu --then $py -B .\scripts\validate_qwen.py --profile qwen3 --model $qwen3 --probe '{bin}\libraries\qwen3\qwen3_generate.exe'

## Llama et Qwen 3.5

Le tokenizer du fork dépend de SentencePiece 0.2.1, SWIG 4.4.1 et PCRE2 10.45. On peut fournir leurs trois archives sources à Bazel via `--distdir=<dossier>` avant `--then`. Bazel contrôle les empreintes épinglées. L’archive SWIG du miroir MacPorts a la même empreinte que celle déclarée dans le fork. [SentencePiece](https://github.com/google/sentencepiece/releases/tag/v0.2.1) · [SWIG](https://distfiles-origin.macports.org/swig/) · [PCRE2 dans le registre Bazel](https://bcr.bazel.build/modules/pcre2/10.45)

Les cibles Windows corrigent la résolution des en-têtes Abseil de SentencePiece et excluent deux en-têtes AVX512-BMM inutilisés par le tokenizer lors de la traduction C vers Zig. [Test Llama](../../libraries/llama/README.md) · [Cible Qwen 3.5](../../libraries/qwen35/README.md)

L’analyse Bazel des deux cibles et la compilation ciblée des bibliothèques SentencePiece et SWIG ont réussi sur Windows x64. La compilation complète et l’exécution des deux lanceurs restent à valider.
