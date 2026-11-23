# Encodeurs T5 et UMT5

Les profils `t5_xxl` et `umt5_xxl` visent le conditionnement texte utilisé par FLUX.1 et Wan. Le graphe ZML calcule l’embedding, l’attention bidirectionnelle à positions relatives, la normalisation RMS et le réseau à GELU avec porte. T5 partage la table de positions de son premier bloc. UMT5 utilise une table par bloc. [T5 v1.1 XXL](https://huggingface.co/google/t5-v1_1-xxl/blob/main/config.json) · [UMT5 XXL](https://huggingface.co/google/umt5-xxl/blob/main/config.json)

Le lanceur `t5_encoder.exe PROFIL [--storage=f16] BACKEND POIDS N ENTREE SORTIE` reçoit deux rangées de `N` valeurs FP32 little-endian. La première contient les identifiants entiers du tokenizer. La seconde contient le masque avec 1 pour un token visible et 0 pour du remplissage. Le lanceur accepte de 1 à 512 tokens et produit `[1,N,4096]` en FP32 pour les profils XXL. Il écrit une nouvelle sortie et journalise le placement sur stderr. Le client fournit les tokens et possède les fichiers d’entrée et de sortie.

`prepare.py` lit les SafeTensors locaux au format Transformers, y compris les index de fichiers segmentés, et extrait les paramètres de l’encodeur. Le format canonique reprend les noms Transformers et utilise FP32 par défaut. Avec `--storage f16` à la conversion et `--storage=f16` au lancement, ZML garde ces poids en FP16 sur le périphérique et effectue les calculs en FP32. Les identifiants et le masque sont préparés avec le tokenizer du checkpoint.

## Validation Windows x64

Les profils `t5_fixture` et `umt5_fixture` reprennent les opérations de leurs profils XXL sur deux blocs réduits. Le script utilise les mêmes poids et les mêmes masques que Transformers, compare deux longueurs de séquence et exécute deux inférences avec chaque modèle chargé.

Les profils réduits passent sur la RTX 3090 avec des poids FP32 et FP16, et sur `cpu:0` avec des poids FP16. Pour 9 et 149 tokens, leur écart absolu maximal face à Transformers est de `2,15e-6`.

Les checkpoints XXL de FLUX et Wan ont été convertis en SafeTensors FP16 avec respectivement 219 et 242 tenseurs de paramètres vérifiés. La référence utilise Transformers 4.57.3 sur CPU, avec les mêmes valeurs de poids et les mêmes entrées que ZML. Le tableau donne l’écart absolu maximal sur les sorties FP32.

| Encodeur | CPU Windows, 9 et 149 tokens | RTX 3090 via `iree_cuda`, 9 et 149 tokens | RTX 3090, 512 tokens |
| --- | ---: | ---: | ---: |
| T5-XXL | `4,05e-6` | `1,34e-5` | `1,08e-4` |
| UMT5-XXL | `3,88e-5` | `1,23e-4` | `3,35e-5` |

Les fichiers de départ sont les distributions FP16 de ComfyUI publiées sous Apache 2.0. Les empreintes des fichiers téléchargés ont été vérifiées.

- [T5-XXL de FLUX, révision 168aff5](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/168aff5/t5xxl_fp16.safetensors) · SHA-256 `6e480b09fae049a72d2a8c5fbccb8d3e92febeb233bbe9dfe7256958a9167635`
- [UMT5-XXL de Wan, révision dfcea77](https://huggingface.co/Comfy-Org/Wan-2.1/blob/dfcea77/split_files/text_encoders/umt5_xxl_fp16.safetensors) · SHA-256 `7b8850f1961e1cf8a77cca4c964a358d303f490833c6c087d0cff4b2f99db2af`

```powershell
$env:ZML_TEMP_ROOT = 'D:\_Project\zml_lib_Temp'
$py = 'D:\ComfyUI\.venv\Scripts\python.exe'
$env:ZML_CPU_PJRT_DLL = (Resolve-Path '..\zml_win\bazel-win-tools\pjrt-cpu\pjrt_cpu.dll').Path
$cpu = @('--@zml//platforms:cuda=false', '--@zml//platforms:rocm=false', '--@zml//platforms:vulkan=false')
& $py -B .\scripts\build.py --target //libraries/t5:t5_encoder -- --repo_contents_cache= --jobs=12 @cpu --then $py -B .\libraries\t5\validate.py --probe '{bin}\libraries\t5\t5_encoder.exe'
```

Pour comparer un checkpoint XXL depuis un fichier local, ajouter `--profile t5_xxl --source $source --storage f16` au comparateur. Utiliser `--profile umt5_xxl` pour le checkpoint Wan. Le comparateur vérifie les noms et les formes face à Transformers, charge les mêmes poids en FP32 pour la référence CPU et teste les deux longueurs avec le même masque d’attention. Le chargement est répété pour chaque longueur et chaque session effectue deux inférences.

Pour la RTX 3090, définir les variables du plugin CUDA et du compilateur IREE, puis exécuter le comparateur. [Configuration CUDA Windows](../botsort/README.md#validation-cuda-sous-windows-x64)

```powershell
$cuda = @('--@zml//platforms:cuda=true', '--@zml//platforms:rocm=false', '--@zml//platforms:vulkan=false')
& $py -B .\scripts\build.py --target //libraries/t5:t5_encoder -- --repo_contents_cache= --jobs=16 --host_compilation_mode=fastbuild @cuda `
  --then $py -B .\libraries\t5\validate.py --probe '{bin}\libraries\t5\t5_encoder.exe' `
  --profile t5_xxl --source $source --storage f16 --backend cuda --expected-device 'RTX 3090'
```

Ajouter `--length 512` au comparateur pour rejouer l’essai à la limite de contexte.

Pour préparer les poids séparément, appeler `prepare.py --probe $probe --profile t5_xxl --source $source --output $weights --storage f16`. L’intégration avec les graphes de diffusion FLUX et Wan et le chargement des variantes FP8 et GGUF restent à faire.
