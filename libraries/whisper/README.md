# Whisper

Le port Zig/ZML fournit l’encodeur et le calcul des logits du prochain token pour Tiny et Large v3 Turbo. Le décodeur applique l’attention causale et l’attention croisée avec les caractéristiques de l’encodeur. Le lanceur Python lit le fichier audio et renvoie le texte et les segments horodatés. [Configuration Large v3 Turbo](https://huggingface.co/openai/whisper-large-v3-turbo/blob/41f01f3/config.json)

`prepare.py --probe $probe --profile large_v3_turbo --source $checkpoint --output $weights` vérifie les noms et formes puis convertit un fichier SafeTensors local de Transformers. Il écrit `encoder.safetensors` et `decoder.safetensors` en FP32 dans un nouveau dossier. Le paramètre `--profile tiny` sélectionne le schéma Tiny.

Les fichiers bruts utilisent FP32 little-endian contigu. Pour Large v3 Turbo, l’encodeur reçoit `[1,128,3000]` et renvoie `[1,1500,1280]`. Le décodeur reçoit cette sortie aplatie immédiatement suivie de `N` identifiants de tokens représentés par des entiers exacts en FP32. Il renvoie les `51 866` logits pour le prochain token. L’appelant possède les buffers et les fichiers.

```powershell
& $probe large_v3_turbo encoder cpu $encoderWeights $melFile $encodedFile
& $probe large_v3_turbo decoder cpu $decoderWeights $packedInput $logitsFile $promptLength
```

## Validation Windows x64

Le profil réduit a donné un écart absolu maximal de `4,77e-7` sur l’encodeur et de `5,97e-8` sur les logits face à Transformers FP32. Large v3 Turbo, avec les poids officiels et un log-mel d’essai synthétique, a donné un écart maximal de `2,04e-3` sur l’encodeur et de `2,80e-4` sur les logits. Les deux tokens gloutons `220` et `3254` sont identiques à la référence. Chaque calcul a vérifié le placement sur `cpu:0`. [Checkpoint officiel](https://huggingface.co/openai/whisper-large-v3-turbo/tree/41f01f3)

```powershell
$env:ZML_TEMP_ROOT = 'D:\_Project\zml_lib_Temp'
$py = 'D:\ComfyUI\.venv\Scripts\python.exe'
$env:ZML_CPU_PJRT_DLL = (Resolve-Path '..\zml_win\bazel-win-tools\pjrt-cpu\pjrt_cpu.dll').Path
$cpu = @('--@zml//platforms:cuda=false', '--@zml//platforms:rocm=false', '--@zml//platforms:vulkan=false')
& $py -B .\scripts\build.py --target //libraries/whisper:whisper_probe -- --jobs=12 @cpu --then $py -B .\libraries\whisper\validate.py --probe '{bin}\libraries\whisper\whisper_probe.exe' --backend cpu --expected-device 'cpu:0'
```

Ajouter `--profile large_v3_turbo --model $model` avec le dossier Transformers local pour rejoindre le profil officiel.

## Transcrire un fichier audio

`transcribe.py` utilise le dossier Transformers local pour le tokenizer, les réglages log-mel et les règles de génération. Il lit l’audio avec SoundFile, mélange les canaux en mono et rééchantillonne à 16 kHz avec SciPy. Il applique la génération gloutonne, les règles de masquage de Transformers et les tokens de timestamp à pas de 20 ms. Avec `--language auto`, le décodeur choisit la langue sur la première fenêtre et garde ce choix pour l’enregistrement. Le mode prend des fenêtres jusqu’à 30 secondes et reporte la fin d’un segment incomplet dans la fenêtre suivante. Les timestamps sont des positions de segments en secondes à partir du début du fichier. [Génération Whisper](https://huggingface.co/docs/transformers/model_doc/whisper#transformers.WhisperForConditionalGeneration.generate)

Le processus Zig conserve les graphes et les poids de l’encodeur et du décodeur pour toute la transcription. Il calcule les clés et valeurs d’attention croisée une fois par fenêtre et les garde sur le périphérique. Le décodeur utilise une capacité fixe pour réutiliser le graphe. Il recalcule l’attention propre du préfixe à chaque token.

```powershell
$env:ZML_TEMP_ROOT = 'D:\_Project\zml_lib_Temp'
& $py -B .\libraries\whisper\transcribe.py --audio $wav --model $model --weights $weights --probe $probe --language fr --backend cuda --expected-device 'RTX 3090'
```

Le JSON sur stdout contient `text`, `segments`, `language`, `duration`, `complete`, `token_windows` et `placement`. Chaque segment contient `start`, `end` et `text`. `complete` passe à `false` si une fenêtre atteint la limite `--max-new-tokens` avant la fin de génération. `placement` donne le backend, la plateforme PJRT, le périphérique et le nombre de fragments vérifiés. Les diagnostics vont sur stderr. Les poids et le fichier audio appartiennent à l’appelant et sont lus depuis leurs chemins locaux. `--workspace` permet à un appelant de placer les fichiers temporaires dans un dossier de tâche existant.

`--runtime-config` lit les champs `probe` et `weights` d’un JSON local en complément des arguments de transcription. Les chemins de ces deux champs sont absolus. Le champ optionnel `expected_cuda_device` restreint le nom du périphérique vérifié en CUDA. Les clients CyTools lisent aussi le champ `launcher` pour localiser ce script.

Le protocole natif `whisper_probe.exe PROFIL transcribe BACKEND ENCODEUR DECODEUR CAPACITE` est synchrone. La commande binaire 0 sur stdin est suivie du tenseur log-mel en FP32 et reçoit un octet 0. La commande 1 est suivie de la position active en U32 puis de `CAPACITE` identifiants U32, avec remplissage par zéro. Elle reçoit les logits FP32 dans l’ordre du vocabulaire. Toutes les valeurs binaires sont little-endian. Fermer stdin libère les ressources.

CyTools AudioStudio expose ce lanceur via `transcribe_audio`. Le réglage `runtime=zml` utilise le fichier local désigné par `CYTOOLS_WHISPER_ZML_CONFIG`. Ce fichier indique les chemins absolus vers le lanceur, le probe et le dossier des poids convertis. Le README de l’application détaille la configuration. Via sa CLI Windows, le WAV de 5,45 s a donné deux segments sur `cpu:0` avec une langue explicite. Le WAV de 35,45 s a donné quatre segments sur la RTX 3090 via `iree_cuda` avec détection de l’anglais.

CyTools PersonaRP utilise le même fichier de configuration pour `transcribe`, la dictée et la transcription des prises de référence. Sa CLI Windows a restitué le même texte et les mêmes horodatages sur `cpu:0` et sur la RTX 3090 via `iree_cuda`. L’export Colab reprend ces sources et utilise PyTorch par défaut. L’inférence ZML sur Colab reste à valider avec un probe et un moteur PJRT Linux.

### Rejouer la validation audio

Fournir un WAV parlé de moins de 30 secondes et le dossier local de Large v3 Turbo avec `model.safetensors`, les configs et le tokenizer. Le comparateur prépare les poids dans son répertoire temporaire et vérifie les tokens, le texte et les timestamps contre `Transformers.generate()`. Il rejoue également un WAV de deux fenêtres avec une seconde occurrence de la parole à 30 secondes. Utiliser `--limit` pour choisir le budget de tokens de l’extrait. [Variables CUDA](../botsort/README.md#validation-cuda-sous-windows-x64)

Sous Windows x64 sur `cpu:0` et sur la RTX 3090 avec `iree_cuda`, Large v3 Turbo a reproduit les 21 tokens, le texte et les timestamps de Transformers 4.57.3 sur un WAV anglais de parole synthétisée de 5,45 s. Le WAV de 35,45 s a reproduit les 42 tokens sur deux fenêtres dans le même processus. Il retrouve les segments `[0; 1,86]`, `[2,72; 4,64]`, `[30; 31,86]` et `[32,72; 34,64]` secondes. Le test réduit du cache a aussi passé deux fenêtres et quatre étapes de décodage en une session sur CPU et CUDA. L’écart maximal des logits de ce test est de `1,20e-7` face à PyTorch.

```powershell
& $py -B .\scripts\build.py --target //libraries/whisper:whisper_probe -- --jobs=12 --host_compilation_mode=fastbuild @cuda `
  --then $py -B .\libraries\whisper\validate_transcription.py --probe '{bin}\libraries\whisper\whisper_probe.exe' `
  --model $model --audio $wav --language en --limit 40 --backend cuda --expected-device 'RTX 3090'
```
