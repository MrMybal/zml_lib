# Phonon 2, 2.5 et 2.5 English

Les trois variantes dérivent de Parakeet TDT v3 et partagent un FastConformer Zig/ZML à 24 blocs et un décodeur à deux couches LSTM. L’attention de l’encodeur emploie les positions relatives. Le réseau joint produit 8 193 scores de token, dont le blanc, et cinq scores de durée. Le lanceur renvoie la transcription ainsi que les horodatages des tokens et des mots. [Fiche Phonon 2](https://huggingface.co/FermionResearch/Phonon-2) · [Fiche Phonon 2.5](https://huggingface.co/FermionResearch/Phonon-2.5)

## Poids

`prepare.py` lit le `model.fermion` déjà extrait par CyTools Subtitle. Il compare le SHA‑256 avec `packed_manifest.json` lorsqu’il est présent. Il décode les poids five-value, Int6, Int8 et FP16, puis écrit des tenseurs FP32. Les 24 compteurs BatchNorm servent à l’entraînement et sont omis dans la conversion. Les fichiers de distribution `.bps.tar.zst` et `.fsq` passent par l’extraction de Fermion ou de CyTools avant cette étape. Les poids de Fermion Research et NVIDIA sont sous CC BY 4.0. [Format Fermion](https://huggingface.co/FermionResearch/Phonon-2/blob/c144cdce269c193be8521f8af7ee92d1d17389ae/fermion_container.py)

```bash
python -B libraries/phonon/prepare.py --model "$model" --validate-payload
python -B libraries/phonon/prepare.py --model "$model" --part encoder --output "$scratch/phonon-encoder.safetensors"
python -B libraries/phonon/prepare.py --model "$model" --part tdt --output "$scratch/phonon-tdt.safetensors"
```

Les checkpoints extraits de Phonon 2 Int6 et de Phonon 2.5 Int6 dans CyTools contiennent chacun 699 tenseurs d’inférence. Leurs manifestes et ces tenseurs ont été vérifiés localement sans écrire de nouveaux poids.

## Transcription

Compiler la cible Bazel `//libraries/phonon:phonon_asr`. Le lanceur Python utilise SoundFile, SciPy, librosa et tokenizers. Il mélange les canaux en mono, rééchantillonne à 16 kHz et prépare 128 bandes log-mel au pas de 10 ms. FastConformer et TDT s’exécutent dans le même processus Zig/ZML. Le décodeur utilise les durées TDT pour avancer dans les trames de 80 ms. Les horodatages des mots sont regroupés à partir des tokens SentencePiece. [Tokenizer Parakeet v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/blob/541d1f99c6b0c3cd0b11a95167540bb8edefd82b/tokenizer.json)

```bash
python -B libraries/phonon/transcribe.py --probe "$probe" --scratch "$scratch" \
  --encoder "$scratch/phonon-encoder.safetensors" --tdt "$scratch/phonon-tdt.safetensors" \
  --audio "$wav" --tokenizer "$processor/tokenizer.json" \
  --backend cuda --expected-device cuda:0 --expected-pjrt cuda
```

Le lanceur écrit le JSON sur stdout. `--chunk-seconds` règle la durée maximale d’un segment entre 5 et 30 secondes. Par défaut, il traite les fichiers de plus de 30 secondes par segments contigus et cherche un passage de faible énergie vers la fin de chaque segment. Si la parole est continue, la coupure peut traverser un mot. Le contexte TDT redémarre à chaque segment. Le report des horodatages sur la durée totale et le champ `segments` permettent de retrouver la position de chaque passage dans le fichier source.

`--overlap-seconds` active les fenêtres avec contexte utilisées par CyTools Subtitle. La moitié du chevauchement est lue de chaque côté du segment et la durée totale de la fenêtre est limitée à 30 secondes. Chaque mot et ses tokens sont conservés dans le segment qui contient le milieu du mot. Les champs `window_start` et `window_end` donnent les bornes effectivement lues.

La commande native `phonon_asr schema` liste les 684 tenseurs FastConformer. Le mode `encode` accepte une matrice FP32 little-endian de log-mels `[T, 128]` et sort `[ceil(T / 8), 1024]`. Le mode `decode` accepte cette matrice et produit la trace JSON des tokens, des trames et des durées. Le mode `transcribe` enchaîne les deux avec les deux checkpoints préparés.

Pour les enregistrements longs, `phonon_asr stream BACKEND ENCODEUR TDT` ouvre une session et charge chaque checkpoint une fois. Chaque requête sur stdin contient un entier non signé U32 little-endian donnant `T`, puis `T × 128` valeurs FP32 little-endian. Une ligne JSON sur stdout donne la trace du segment. La fermeture de stdin libère la session et les poids. Le lanceur vérifie le placement de l’encodeur et du décodeur pour chaque segment.

En dictée directe, le lanceur Python utilise `--live` à la place de `--audio`. Il préchauffe la session puis émet un événement `ready`. Chaque ligne `{"audio": "chemin.wav"}` reçoit une ligne `transcription` avec le texte, les mots, les tokens, les horodatages et le placement. À la fermeture de stdin, `closed` confirme le nombre de fenêtres traitées. Le binaire doit exposer `phonon_asr stream-live`.

Le mode direct réutilise un encodeur compilé pour 2 000 trames log-mel. Le masque exclut le remplissage des clés d’attention et des entrées des convolutions temporelles. Le décodeur TDT traite uniquement les trames qui correspondent à l’audio. `validate_live.py` compare les sorties de l’encodeur aux fenêtres de taille réelle, vérifie les tokens et horodatages sur plusieurs durées et mesure le temps de réponse après préchauffage.

```bash
python -B libraries/phonon/validate_live.py --probe "$probe" --scratch "$scratch" \
  --encoder "$encoder" --tdt "$tdt" --tokenizer "$tokenizer" --audio "$wav" \
  --backend cuda --expected-pjrt cuda --expected-device cuda:0 \
  --output "$report"
```

Le test attend un extrait de parole de plus de 1,25 s et d’au plus 20 s, une RMSE de l’encodeur d’au plus 0,001 et une égalité exacte des transcriptions et des horodatages. Sous Windows, le PJRT CUDA du fork épinglé s’appelle `iree_cuda`.

Sur Colab L4, quatre fenêtres de 1,21 s à 2 s de Phonon 2.5 ont reproduit le texte, les tokens et leurs horodatages de l’inférence ZML sans remplissage. La RMSE maximale de l’encodeur était de 1,75 × 10⁻⁸ pour les longueurs de 121 et 122 trames. Une session a pris 18,05 s à préchauffer, puis les fenêtres ont répondu en 59 à 68 ms sur `cuda:0`. Le test a aussi vérifié le rejet d’un périphérique incorrect et la fermeture de la session.

Le binaire Linux autonome a traité une fenêtre de 20 s en 160 ms après préchauffage et a reproduit la transcription et les horodatages de l’inférence sans remplissage. Le pic relevé pendant les essais Phonon était de 6 062 Mio. Les extraits, les résultats et les empreintes figurent dans `validation-output/phonon-live-colab-l4.tar.gz`.

`validate_long.py` reprend les options de transcription. Le test compare le texte, les tokens, les mots et leurs horodatages à ceux des mêmes segments exécutés dans des processus indépendants. L’option `--output` conserve la transcription JSON du fichier entier.

Sur Colab L4, Phonon 2 a traité 79,86 s en anglais et Phonon 2.5 a traité 70 s en français. Les deux fichiers ont été construits en répétant les extraits courts avec des pauses. Chaque transcription a utilisé trois segments et un chargement des deux checkpoints. Le texte, les tokens et les horodatages sont identiques à ceux des segments exécutés indépendamment. Les 204 mots attendus en anglais et les 50 mots attendus en français ont été retrouvés. En anglais, le modèle a écrit « Mr. » pour « Mister ». Les deux essais courts ont également reproduit les JSON de référence.

### CyTools Subtitle

L’opération `transcribe` accepte `"asr_engine": "zml"`. Le moteur Fermion CPU reste le choix des requêtes existantes. Le fichier JSON désigné par `CYTOOLS_PHONON_ZML_CONFIG` associe les IDs de modèles aux poids préparés et fixe le lanceur, le backend, la plateforme PJRT et le périphérique. Consultez la section « Optional Phonon ZML runtime » du README de l’application pour la configuration.

Subtitle prépare un WAV mono à 16 kHz, transmet les réglages de fenêtre et de chevauchement et produit ses exports JSON, SRT et WebVTT. Le JSON conserve les horodatages des mots et le placement vérifié. Sous Windows, `app/test_phonon_zml.py` vérifie le routage via la CLI publique, les exports, les erreurs et l’arrêt des sous-processus avec un lanceur simulé. L’inférence ZML native sous Windows reste à valider.

Sur Colab L4, le raccordement de Subtitle a transcrit un enregistrement français de 17 secondes en trois fenêtres. La taille de segment était réglée à 6,5 secondes et le chevauchement à 2 secondes. Les six mots et leurs horodatages correspondent exactement aux mêmes fenêtres exécutées indépendamment. L’inférence a utilisé le PJRT `cuda` sur `cuda:0`. Le checkpoint public Phonon 2.5 extrait avait le même SHA-256 que celui de CyTools.

### CyTools AIChat

La transcription de fichiers et la dictée en direct via la CLI ou MCP acceptent `"asr_engine": "zml"`. Avec un binaire qui expose `stream-live`, activez l’option d’affichage des mots pendant la parole dans les réglages ZML d’AIChat. Le texte arrive dans le champ de saisie pendant l’enregistrement. Ce réglage ZML est désactivé par défaut, y compris pour les configurations existantes. PyTorch CPU reste le moteur par défaut.

AIChat reprend `CYTOOLS_PHONON_ZML_CONFIG` et renvoie le placement vérifié. Les tests Windows utilisent un lanceur simulé pour vérifier le choix du modèle, les réglages, le routage du micro et l’arrêt de l’arbre de processus.

Sur Colab L4, l’adaptateur AIChat a transcrit trois fenêtres via une session Phonon 2.5, avec le même texte que le lanceur direct. Le préchauffage a pris 16,79 s, puis le temps de réponse des fenêtres était de 91 à 129 ms. L’annulation a libéré les processus et la réservation de mémoire. Le fichier de 40 s en mode segmenté a également reproduit son rapport de référence à l’identique.

La CLI AIChat de Windows x64 a aussi transcrit deux fois un extrait français de 2 secondes et une fois un enregistrement de 40 secondes avec Phonon 2.5. Le lanceur de validation déclenchait l’inférence sur Colab L4 pour les deux fichiers synthétiques identifiés par SHA-256. Le texte, la durée et le placement `cuda:0` correspondent aux inférences ZML directes. Le fichier long a produit 14 mots en deux segments. Les tâches CLI ont libéré leurs fichiers temporaires. Les WAV et les rapports figurent dans `validation-output/phonon-aichat-colab-l4.tar.gz`. La validation CUDA native dans AIChat sous Windows reste à faire.

Pour reproduire la comparaison avec la CLI, conservez le JSON de l’inférence directe décrit plus haut. Utilisez le même fichier audio, le même modèle et le même backend pour les deux appels. Le script vérifie aussi la fermeture des tâches et supprime son dossier temporaire.

```powershell
$app = 'D:\AI\CyTools\CyTools_AIChat'
$py = "$app\runtime\python\Scripts\python.exe"
& $py -B .\libraries\phonon\validate_aichat.py --tool "$app\app\tool.py" `
  --config $env:CYTOOLS_PHONON_ZML_CONFIG --model phonon-2.5 --audio $wav `
  --reference $referenceJson --scratch 'D:\AI\zml_lib_Temp' `
  --report '.\validation-output\phonon-aichat.json' --repeats 2
```

### Comparaison avec PyTorch

Pour un extrait court, le script `validate_asr.py` reprend les options du lanceur et ajoute `--config "$processor/config.json" --reference-device cuda:0`. Il compare l’extraction log-mel et l’encodeur, puis le texte, les tokens et leurs horodatages. Avec `--expected-text`, il calcule aussi le taux d’erreur de mots par rapport au texte attendu. La référence utilise Transformers 5.6.0 et PyTorch FP32. Sur Linux CUDA, les lanceurs appliquent au processus ZML des réglages qui écartent Triton GEMM et cuBLASLt, sauf si l’appelant fournit `XLA_FLAGS`.

Les essais ont utilisé les checkpoints CyTools et le checkpoint public Phonon 2.5 English face à PyTorch CUDA `2.10.0+cu128` sur Colab L4, avec ZML `0bb81d1957fb6438faddd1751566ad73eed97914`. Les séquences de tokens et leurs horodatages correspondent exactement à la référence. Le taux d’erreur de mots est de 0 % après retrait de la ponctuation et normalisation de la casse. Les journaux et les sorties JSON de la validation Subtitle et de la variante English sont dans `validation-output/phonon-subtitle-colab-l4.tar.gz`. [Extrait anglais LibriSpeech](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy) · [Checkpoint Phonon 2.5 English](https://huggingface.co/FermionResearch/Phonon-2.5-English)

| Checkpoint | Extrait | RMSE log-mel | RMSE FastConformer | Écart maximal des horodatages |
| --- | --- | ---: | ---: | ---: |
| Phonon 2 | Anglais, 5,855 s | 2,83e-6 | 2,91e-8 | 0 ms |
| Phonon 2.5 | Anglais, 5,855 s | 2,83e-6 | 3,53e-8 | 0 ms |
| Phonon 2.5 | Français synthétisé, 2 s | 5,83e-7 | 1,48e-8 | 0 ms |
| Phonon 2.5 English | Anglais, 5,855 s | 2,83e-6 | 3,35e-8 | 0 ms |

Sous Windows x64, la cible peut être construite et testée dans un même espace temporaire.

```powershell
$env:ZML_TEMP_ROOT = 'D:\AI\zml_lib_Temp'
$flags = @('--@zml//platforms:cuda=true', '--@zml//platforms:rocm=false', '--@zml//platforms:vulkan=false')
& $py -B .\scripts\build.py --target //libraries/phonon:phonon_asr -- --jobs=12 @flags `
  --then $py -B .\libraries\phonon\validate_asr.py --probe '{bin}\libraries\phonon\phonon_asr.exe' `
  --scratch $env:ZML_TEMP_ROOT --encoder $encoder --tdt $tdt --audio $wav `
  --tokenizer "$processor\tokenizer.json" --config "$processor\config.json" `
  --backend cuda --expected-device 'RTX 3090' --expected-pjrt iree_cuda --reference-device cuda:0
```

## Pas TDT

`phonon_tdt_probe BACKEND POIDS ENTRÉE SORTIE [ENTRÉE SORTIE ...]` traite plusieurs appels avec le même modèle chargé. Le fichier d’entrée comporte un identifiant du dernier token, les 1 024 features d’une trame, puis `h0, c0, h1, c1` de 640 valeurs chacun. Le fichier de sortie contient 8 193 scores de token, cinq scores de durée puis les quatre nouveaux états. Les valeurs sont FP32 little-endian. Le moteur vérifie le placement avant la copie. Le client conserve les états précédents si le blanc est choisi.

```bash
python -B libraries/phonon/validate_tdt.py --model "$model" --probe "$probe" \
  --scratch "$scratch" --backend cuda --expected-device cuda:0 --expected-pjrt cuda \
  --reference-device cuda:0
```

Le script prépare le poids TDT dans son espace temporaire et compare les scores de token, les scores de durée et les états LSTM avec PyTorch. Chaque RMSE doit rester sous 0,001 et les deux `argmax` doivent correspondre. Sous Windows x64, construire `//libraries/phonon:phonon_tdt_probe` via `scripts/build.py` avec le moteur épinglé, sélectionner le PJRT `iree_cuda` pour CUDA et utiliser `D:\AI\zml_lib_Temp` pour le build et la validation.

Trois appels ont été comparés pour chaque fichier CyTools sur Colab L4, avec des trames d’encodeur synthétiques et les poids réels. Les 15 tenseurs TDT sont identiques dans ces deux checkpoints. La RMSE maximale face à PyTorch CUDA a été de `3,83e-5` sur les scores de token, `1,22e-5` sur les durées et `1,68e-7` sur les états. Les argmax de token et de durée correspondent à chaque appel. Le moteur a confirmé le placement `cuda:0` avant la copie.
