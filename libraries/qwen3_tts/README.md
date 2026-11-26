# Qwen3‑TTS 12 Hz

Les graphes Zig/ZML couvrent la projection du texte, les tables de codes, les 28 couches du Talker et les 5 couches qui prédisent les 15 autres groupes de codes par trame. Les profils 0.6B et 1.7B partagent les couches du port Qwen, avec des dimensions et des checkpoints distincts. Les poids sont stockés en FP16 et les calculs s’effectuent en FP32. [Implémentation Qwen](https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/core/models/modeling_qwen3_tts.py)

Le lanceur compose le prompt de CustomVoice, VoiceDesign ou Base et calcule les scores de code avec ZML. CustomVoice et VoiceDesign utilisent le choix glouton si `--seed` est omis. Base échantillonne les codes selon les réglages de Qwen. Le Talker recalcule le préfixe causal à chaque trame. Il réserve d’abord jusqu’à 64 trames de génération, puis agrandit le contexte si la synthèse continue. Le décodeur audio ZML partagé reconstruit un WAV mono PCM 16 bits à 24 kHz à partir des 16 codes par trame. Il exécute la quantification résiduelle inverse, le transformeur causal à fenêtre glissante et les couches de synthèse audio. [Codec 12 Hz officiel, Apache 2.0](https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz/tree/7dd38ad4e9bad454aae9cd937d0cd577604fe229)

## Conversion et génération

Utiliser le dossier local du checkpoint avec son fichier `config.json`, ses fichiers SafeTensors et son tokenizer texte. Le convertisseur contrôle les noms, dimensions et types du checkpoint officiel contre le schéma du probe. Il stocke les poids des étapes `text`, `codec`, `talker` et `predictor` en SafeTensors FP16. Les fichiers préparés peuvent être réutilisés entre plusieurs appels.

La conversion utilise PyTorch et `safetensors`. Le lanceur natif utilise `numpy`, `safetensors` et `transformers` pour charger les tables et le tokenizer texte. Le clonage utilise aussi `soundfile`, `librosa` et `scipy` pour le prétraitement audio. `qwen-tts` et PyTorch servent à la comparaison de référence.

```bash
for stage in text codec talker predictor; do
  python -B libraries/qwen3_tts/prepare.py --probe "$probe" --profile qwen3_tts_0b6 \
    --model "$model" --stage "$stage" --output "$prepared/$stage.safetensors"
done
python -B libraries/qwen3_tts/prepare.py --probe "$probe" --stage waveform \
  --model "$speech_tokenizer" --output "$prepared/waveform.safetensors"
python -B libraries/qwen3_tts/generate.py --probe "$probe" --profile qwen3_tts_0b6 \
  --model "$model" --prepared "$prepared" --scratch "$scratch" --text "Bonjour." \
  --language french --speaker serena --backend cuda --expected-device cuda:0 \
  --expected-pjrt cuda --waveform-weights "$prepared/waveform.safetensors" --output bonjour.wav
```

`$speech_tokenizer` désigne le dossier local du codec officiel, par exemple `$model/speech_tokenizer`. Le convertisseur vérifie 269 tenseurs de décodage et les garde en FP32. Pour VoiceDesign, choisir le profil 1.7B et donner une `--instruction` de voix. Pour sortir les codes uniquement, indiquer un fichier `.npy` et omettre les deux options de décodeur. L’option `--speech-tokenizer` exécute le décodeur de référence avec `qwen-tts==0.1.1`. La réponse JSON indique le nombre de trames, la présence d’une fin de génération et le moteur qui a produit le WAV.

Le protocole `qwen3_tts_probe waveform BACKEND POIDS TRAMES stdin` accepte de 1 à 325 trames de 16 codes entiers de 0 à 2 047, encodés en FP32 little-endian. Chaque requête commence par un octet zéro et renvoie `TRAMES × 1920` échantillons FP32 bruts. Le lanceur découpe les séquences longues en blocs de 300 trames, fournit jusqu’à 25 trames de contexte et retire leurs échantillons avant la concaténation.

## Clonage à partir d’un enregistrement avec Base

Base prend en charge les deux modes officiels de clonage `x_vector_only` et ICL. Avec `--reference-audio`, le lanceur lit un fichier audio local, le mélange en mono et le rééchantillonne à 24 kHz avec SoundFile et librosa. SciPy calcule le spectre utilisé pour les 128 bandes log-mel. Le graphe ZML ECAPA-TDNN extrait ensuite un vecteur FP32 de 1 024 ou 2 048 valeurs. Le Talker l’insère entre les balises de langue et le code de remplissage. En ICL, le modèle reçoit aussi la transcription et les codes acoustiques de l’enregistrement. [Clonage officiel Qwen](https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/inference/qwen3_tts_model.py#L323-L342)

Préparez les cinq étapes ci-dessous depuis le checkpoint Base. `speaker` vérifie les 76 tenseurs et les convertit en FP32. Le décodeur WAV réutilise le checkpoint préparé avec `--stage waveform`. `--seed` rend reproductibles les tirages de l’échantillonnage Base. Le lanceur applique top‑k 50, température 0,9 et une pénalité de répétition de 1,05 au premier groupe. [Base 0.6B](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/tree/5d83992436eae1d760afd27aff78a71d676296fc) · [Base 1.7B](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/tree/fd4b254389122332181a7c3db7f27e918eec64e3)

```bash
for stage in speaker text codec talker predictor; do
  python -B libraries/qwen3_tts/prepare.py --probe "$probe" --profile "$profile" \
    --model "$base" --stage "$stage" --output "$prepared/$stage.safetensors"
done
python -B libraries/qwen3_tts/generate.py --probe "$probe" --profile "$profile" \
  --model "$base" --prepared "$prepared" --scratch "$scratch" \
  --reference-audio "$reference" --text "Bienvenue dans notre studio." --language french --seed 42 \
  --backend cuda --expected-device cuda:0 --expected-pjrt cuda \
  --waveform-weights "$waveform_weights" --output clone.wav
```

Le probe `qwen3_tts_probe PROFIL speaker BACKEND POIDS TRAMES stdin` prend une matrice FP32 little-endian `[TRAMES, 128]` après un octet zéro et renvoie un vecteur FP32 `[1, H]`. Il accepte de 5 à 2 812 trames, soit jusqu’à 30 secondes de référence. L’empreinte de locuteur, les codes et le WAV sont calculés sur le backend demandé par le lanceur.

Pour ICL, préparez les 159 tenseurs du modèle Mimi à partir du même checkpoint 12 Hz que le décodeur WAV. Ajoutez `--audio-encoder-weights` et `--reference-text` à la commande de génération ou de validation. Le modèle calcule 16 codes par groupe de 1 920 échantillons, avec arrondi au groupe supérieur pour un dernier groupe partiel. Le décodeur reçoit les codes de référence suivis des codes générés. Le WAV exporté commence après les échantillons associés à la référence. [Encodeur Mimi dans Transformers 4.57.3](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/mimi/modeling_mimi.py)

```bash
python -B libraries/qwen3_tts/prepare.py --probe "$probe" --stage audio_encoder \
  --model "$speech_tokenizer" --output "$prepared/audio_encoder.safetensors"
python -B libraries/qwen3_tts/generate.py --probe "$probe" --profile "$profile" \
  --model "$base" --prepared "$prepared" --scratch "$scratch" \
  --reference-audio "$reference" --reference-text "$transcript" \
  --audio-encoder-weights "$prepared/audio_encoder.safetensors" \
  --text "Merci beaucoup." --language french --seed 42 --max-frames 64 \
  --backend cuda --expected-device cuda:0 --expected-pjrt cuda \
  --waveform-weights "$waveform_weights" --output clone-icl.wav
```

Le protocole `qwen3_tts_probe audio_encoder BACKEND POIDS ÉCHANTILLONS stdin` accepte un octet zéro suivi du signal mono 24 kHz en FP32 little-endian. Il renvoie `[ceil(ÉCHANTILLONS / 1920), 16]` identifiants encodés en FP32 little-endian. `validate_clone.py` compare en ICL les codes de référence avant de vérifier la synthèse.

`validate_clone.py` utilise le checkpoint officiel Base et le codec officiel avec `qwen-tts==0.1.1`. Il compare le prétraitement, l’empreinte du locuteur, les embeddings du prompt, les codes de synthèse et le WAV. Le test synchronise les tirages sur les distributions de probabilités calculées indépendamment par les deux moteurs. Il exige des codes identiques et une RMSE d’au plus 0,001 pour l’empreinte, le prompt et le WAV. `--alternate-audio` vérifie l’effet d’un deuxième enregistrement sur l’empreinte.

```bash
python -B libraries/qwen3_tts/validate_clone.py --probe "$probe" --profile "$profile" \
  --model "$base" --prepared "$prepared" --scratch "$scratch" --codec "$speech_tokenizer" \
  --waveform-weights "$waveform_weights" --reference-audio "$reference" \
  --text "Bienvenue dans notre studio." --language french --seed 42 \
  --backend cuda --expected-device cuda:0 --expected-pjrt cuda --reference-device cuda:0
```

Sur Colab L4, les tests ont utilisé deux voix synthétiques issues des essais CustomVoice et VoiceDesign, de 1,28 s et 1,44 s. Les deux profils Base ont reproduit exactement les codes de PyTorch CUDA `2.10.0+cu128` avec la graine 42 et les mêmes tirages. La RMSE maximale des log-mels est de 4,92 × 10⁻⁶. Les deux enregistrements ont produit des empreintes distinctes et chaque appel répété a reproduit exactement son empreinte.

| Profil | Texte demandé | Trames | Durée du WAV | RMSE de l’empreinte principale | RMSE du WAV |
| --- | --- | ---: | ---: | ---: | ---: |
| Base 0.6B | Bienvenue dans notre studio. | 22 | 1,76 s | 2,09 × 10⁻⁷ | 3,53 × 10⁻⁸ |
| Base 1.7B | Bonjour. | 11 | 0,88 s | 3,89 × 10⁻⁷ | 6,40 × 10⁻⁸ |

Le rapport `validation-output/qwen3-tts-base-colab-l4.tar.gz` contient les références, les WAV de chaque moteur et les mesures. Sous Windows, un essai avec un lanceur simulé a vérifié le passage d’une source stéréo à 44,1 kHz vers les log-mels et l’insertion de l’empreinte dans le prompt. L’inférence native Windows reste à valider.

En ICL, ZML et PyTorch CUDA ont produit exactement les mêmes codes acoustiques pour les deux enregistrements, y compris lorsqu’il manque un échantillon dans la dernière trame. Les codes générés sont également identiques avec la graine 42 et les mêmes tirages. Les durées ci-dessous correspondent à la voix générée après retrait du signal de référence.

| Profil | Trames de référence | Trames générées | Durée du WAV | RMSE du prompt | RMSE du signal audio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base 0.6B | 16 | 23 | 1,84 s | 3,49 × 10⁻⁸ | 4,39 × 10⁻⁸ |
| Base 1.7B | 18 | 15 | 1,20 s | 6,35 × 10⁻⁸ | 1,49 × 10⁻⁷ |

## AudioStudio et PersonaRP

Faites pointer `CYTOOLS_QWEN3_TTS_ZML_CONFIG` vers un fichier JSON local. L’interface, la CLI et MCP suivent la même sélection de moteur dans le worker de synthèse. AudioStudio accepte aussi `runtime=zml` ou `runtime=torch` pour `voice_qwen3tts`. Sans configuration ZML, la sélection automatique utilise le moteur PyTorch. Les applications démarrent leur environnement de worker habituel. Le script `cytools_bridge.py` appelle le lanceur ZML avec le modèle déjà sélectionné par l’application et utilise le dossier de la tâche pour ses fichiers temporaires.

```json
{
  "bridge": "D:/AI/zml_lib/libraries/qwen3_tts/cytools_bridge.py",
  "probe": "D:/models/zml/qwen3_tts_probe.exe",
  "waveform_weights": "D:/models/zml/waveform.safetensors",
  "audio_encoder_weights": "D:/models/zml/audio_encoder.safetensors",
  "prepared": {"0.6B-Base": "D:/models/zml/0.6B-Base"},
  "devices": {"cuda": {"device": "RTX 3090", "pjrt": "iree_cuda"}}
}
```

Ajoutez à `prepared` chaque variante à utiliser, avec ses propres poids convertis. Le chemin optionnel `python` permet de choisir l’interpréteur qui possède les dépendances du lanceur. Pour CPU, renseignez `devices.cpu` avec le périphérique `cpu:0` et le PJRT correspondant. Le graphe utilise FP32 et accepte au plus 1 024 positions de contexte, texte, référence et codes compris. PersonaRP rend les groupes de répliques un passage à la fois avec ZML. Sous Windows, utilisez un dossier de tâches sur D: et un binaire construit avec le fork épinglé.

Sur Colab L4, le worker AudioStudio a synthétisé avec les deux CustomVoice, Base 0.6B en ICL et Base 1.7B par empreinte seule. PersonaRP a synthétisé avec VoiceDesign 1.7B et traité un lot avec Base 1.7B en ICL. Pour les deux essais ICL, les WAV des workers et du lanceur direct diffèrent au plus d’une unité PCM. Les deux WAV CustomVoice contiennent exactement les 2 400 échantillons de silence demandés. Les tests de régression des deux applications ont passé sous Windows sans GPU. La compilation et l’inférence CUDA natives Windows restent à valider. Les mesures et extraits WAV sont conservés dans le rapport local `validation-output/qwen3-tts-icl-colab-l4.tar.gz`. [CustomVoice 1.7B](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/tree/0c0e3051f131929182e2c023b9537f8b1c68adfe)

## Validation

Le script compare deux appels dans le même processus pour chacune des quatre opérations, avec le même checkpoint FP16 lu en FP32 par les deux moteurs. Sur une cible Linux compilée, exécuter la commande suivante. Pour le petit profil de test, omettre `--profile` et `--model`.

Sur CUDA Linux, les lanceurs Python règlent `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false --xla_gpu_enable_cublaslt=false` et `NVIDIA_TF32_OVERRIDE=0` pour le processus ZML, sauf si l’appelant a déjà fourni ces variables. Utiliser les mêmes réglages si le probe est appelé directement pour reproduire les mesures.

```bash
python -B libraries/qwen3_tts/validate.py --probe "$probe" --scratch "$scratch" \
  --profile qwen3_tts_0b6 --model "$model" --backend cuda \
  --expected-device cuda:0 --expected-pjrt cuda --reference-device cuda:0
```

Les quatre opérations ont passé deux appels face à PyTorch CUDA `2.10.0+cu128` sur la L4, avec le moteur ZML `0bb81d1957fb6438faddd1751566ad73eed97914`. Le tableau donne les RMSE maximales par opération. [Checkpoint 0.6B CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice/tree/85e237c12c027371202489a0ec509ded67b5e4b5) · [Checkpoint 1.7B VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign/tree/5ecdb67327fd37bb2e042aab12ff7391903235d3)

| Profil | Projection texte | Tables de codes | Talker | Prédicteur |
| --- | ---: | ---: | ---: | ---: |
| 0.6B CustomVoice | 3,41e-9 | 0 | 2,15e-5 | 1,00e-5 |
| 1.7B VoiceDesign | 2,18e-9 | 0 | 2,47e-5 | 1,99e-5 |

Sur la même L4, CustomVoice 0.6B a généré « Bonjour. » avec la voix Serena en 16 trames. VoiceDesign 1.7B a généré « Bonjour, bienvenue. » avec une instruction de ton en 25 trames. Les deux générations ont atteint leur fin. Le décodeur officiel a produit des WAV mono PCM 16 bits à 24 kHz de 1,28 s et 2,00 s. Les extraits et les journaux sont conservés dans `validation-output`.

`validate_waveform.py` vérifie le décodeur ZML face à `qwen-tts==0.1.1` en FP32. Par défaut, les fixtures de 1, 73 et 301 trames couvrent les bornes du graphe, la fenêtre d’attention et la jonction entre blocs. Le test contrôle le placement, le nombre exact d’échantillons, la stabilité des appels répétés et une RMSE d’au plus 0,001. Avec `--codes`, il compare aussi les codes produits par le Talker.

Sur Colab L4, les trois fixtures ont reproduit le nombre d’échantillons de PyTorch CUDA `2.10.0+cu128`. La RMSE maximale est de 5,58 × 10⁻⁷. Les appels répétés dans une même session ont rendu exactement les mêmes échantillons. Sous Windows, un lanceur simulé a vérifié les deux jonctions d’une séquence de 601 trames et l’export PCM. Le décodeur natif Windows reste à valider.

```bash
python -B libraries/qwen3_tts/validate_waveform.py --probe "$probe" --scratch "$scratch" \
  --model "$speech_tokenizer" --waveform-weights "$prepared/waveform.safetensors" \
  --backend cuda --expected-device cuda:0 --expected-pjrt cuda --reference-device cuda:0
```

Les deux profils ont aussi produit un WAV avec `--waveform-weights` et atteint leur fin de génération sur `cuda:0`. La comparaison prend les codes émis par le Talker et les décode dans les deux moteurs. Les deux checkpoints de synthèse publient la même empreinte SHA-256 pour le codec. Les extraits, les codes et les mesures sont conservés dans `validation-output/qwen3-tts-waveform-colab-l4.tar.gz`.

| Profil | Texte demandé | Trames | Durée du WAV | RMSE du codec |
| --- | --- | ---: | ---: | ---: |
| 0.6B CustomVoice · Serena | Bonjour. | 16 | 1,28 s | 3,92 × 10⁻⁸ |
| 1.7B VoiceDesign · voix chaleureuse | Bonjour, bienvenue. | 18 | 1,44 s | 1,60 × 10⁻⁷ |

Sous Windows x64, lancer ces commandes PowerShell depuis le dépôt pour compiler le probe. Le script utilise le moteur Windows épinglé et nettoie son répertoire de build.

```powershell
$env:ZML_TEMP_ROOT = 'D:\AI\zml_lib_Temp'
python -B scripts/build.py --engine 'D:\AI\zml_win' --target //libraries/qwen3_tts:qwen3_tts_probe -- `
  --@zml//platforms:cpu=false --@zml//platforms:cuda=true `
  --@zml//platforms:rocm=false --@zml//platforms:vulkan=false
```

Les scripts de validation prennent `--expected-pjrt iree_cuda` et le nom du périphérique exact rapporté par PJRT avec `--expected-device`. Les chemins d’entrée et de sortie du probe sont fournis par l’appelant. Les buffers sont contigus et FP32 little-endian. En mode `stdin`, chaque requête commence par un octet zéro, puis le buffer d’entrée de taille fixe. Le probe écrit le buffer de sortie complet sur stdout et le placement sur stderr. Fermer stdin libère le modèle.

La largeur `H` vaut 1 024 ou 2 048 selon le profil. `N` est le nombre de lignes, entre 1 et 1 024. Le format de chaque opération est le suivant.

| Opération | Entrée | Sortie |
| --- | --- | --- |
| `text` | `[N, 1]` identifiants de tokens texte | `[N, H]` embeddings projetés |
| `codec` | `[16, 1]` identifiants de code, dans l’ordre des groupes | `[16, H]` embeddings de code |
| `talker` | `[N, H]` embeddings ordonnés du contexte | `[N, H + 3072]` états normalisés suivis des scores |
| `predictor` | `[16, H]` état du Talker puis embeddings des groupes 0 à 14 | `[15, 2048]` scores des groupes 1 à 15 |
