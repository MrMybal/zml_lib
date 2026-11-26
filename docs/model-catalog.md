# Catalogue local des modèles ZML

`scripts/model_catalog.py` donne une entrée commune aux portages existants. Il reconnaît un profil à partir du format, des clés, des dimensions et de la précision des tenseurs. Le nom du fichier peut changer. Le profil identifié définit le graphe à utiliser et le SHA-256 identifie les poids d’origine.

| Bibliothèque | Profils | Entrées |
| --- | --- | --- |
| Real-ESRGAN | `RealESRGAN_x4plus`, `RealESRGAN_x4plus_anime_6B`, `RealESRGAN_x2plus` | Checkpoint PyTorch avec `params_ema` ou `params`, RRDBNet au format séquentiel ESRGAN, SafeTensors RRDBNet |
| Stable Diffusion 1.5 | `sd15`, `sd15-inpaint` | Checkpoint combiné CompVis `.ckpt` ou `.safetensors`, dossier FP32 déjà préparé |
| CLIP | `clip_l` | Encodeur texte CLIP-L SafeTensors au format Transformers, préparé en FP32 |
| T5 | `t5_xxl` | Encodeur T5 v1.1 XXL SafeTensors au format Transformers, préparé en FP16 |
| FLUX.1 | `flux1_vae` | Décodeur VAE à 16 canaux latents, extrait du checkpoint `ae.safetensors` et préparé en FP32 |

Pour Real-ESRGAN, le catalogue vérifie le schéma RRDBNet complet et retrouve le profil x4plus pour les dérivés à 23 blocs compatibles. Les variantes d’entraînement peuvent partager ce profil. Pour Stable Diffusion, il distingue notamment le U-Net à quatre canaux de celui à neuf canaux. Le convertisseur vérifie ensuite toutes les clés et dimensions requises par ZML.

`identify` inspecte les en-têtes SafeTensors et lit les checkpoints PyTorch avec `weights_only=True` et des tenseurs sur le périphérique `meta`. `prepare` appelle le convertisseur existant. Il écrit un nouveau résultat dans le cache choisi par l’appelant et conserve le SHA-256 du checkpoint. La clé du cache tient compte du profil, de la précision, du format d’origine, des fichiers du convertisseur et de la sonde native lorsqu’elle fournit le schéma. Les fichiers préparés sont vérifiés par SHA-256 avant réutilisation. Une entrée altérée produit une erreur.

## Utilisation sous Windows

```powershell
$py = 'D:\AI\ComfyUI\.venv\Scripts\python.exe'
$checkpoint = 'D:\AI\ComfyUI\models\upscale_models\RealESRGAN_x4plus.pth'
$cache = 'D:\models\zml-cache'
& $py -B .\scripts\model_catalog.py list
& $py -B .\scripts\model_catalog.py scan 'D:\AI\ComfyUI\models' 'D:\AI\Comfy-Desktop\ComfyUI-Shared\models'
& $py -B .\scripts\model_catalog.py identify $checkpoint
& $py -B .\scripts\model_catalog.py prepare $checkpoint --cache-dir $cache
```

Pour préparer SD 1.5 ou Inpainting, ajouter `--bin-dir $bin`, où `$bin` contient `libraries\stable_diffusion\sd15_probe.exe`. Le dossier préparé est utilisable par les lanceurs existants avec leurs arguments habituels pour le tokenizer et le périphérique. [Build et lanceurs Stable Diffusion](../libraries/stable_diffusion/README.md)

Pour préparer les composants autonomes, ajouter aussi `--bin-dir $bin`. Le catalogue interroge `clip_text_probe.exe`, `t5_encoder.exe` ou `flux1_probe.exe` pour vérifier chaque clé et dimension attendue. Le résultat est un fichier SafeTensors à fournir au lanceur correspondant. T5-XXL conserve ses poids en FP16 et calcule en FP32. Le VAE fournit le décodage latent vers RGB.

Chaque commande renvoie du JSON sur stdout et les diagnostics sur stderr. `scan` donne le chemin relatif, la taille, l’état et, selon le format, les types et le nombre de tenseurs. `partial_download` signale un fichier `.part` ignoré jusqu’à la fin du téléchargement. `unsupported` signifie que le catalogue n’a pas de profil de détection pour ce fichier. `unreadable` signale une erreur de lecture. `identify` renvoie `needs_preparation` ou `ready`. `prepare` renvoie le chemin utilisable par le lanceur et l’état `prepared`, `cached` ou `ready`. `ready` indique que le fichier présente la signature d’un format préparé. La validation numérique relève des tests du modèle.

## Bibliothèque partagée ComfyUI

Le scan de `ComfyUI-Shared/models` a relevé 32 fichiers de poids terminés et 7 téléchargements en cours. Le scan des deux dossiers donne l’état actuel par fichier et complète l’[inventaire des adaptateurs ComfyUI et CyTools](model-adapters.json).

| Famille repérée | Travail correspondant dans ZML |
| --- | --- |
| CLIP-L, T5-XXL, VAE FLUX.1 | Raccordés au catalogue et aux convertisseurs des portages existants. |
| FLUX.1 Canny Dev | Profil ZML et opération ImageStudio avec une entrée de contours et une projection à 128 valeurs. Le convertisseur prépare un bundle regroupant les quatre checkpoints. [Portage FLUX.1](../libraries/flux1/README.md#flux1-canny-dev) |
| Qwen Image Edit 2511 | Débruiteur FP8 E4M3FN à échelles et VAE pour images fixes, avec réutilisation de Qwen2.5-VL. Les commandes de préparation et de comparaison à PyTorch sont dans le dossier de la bibliothèque. [Portage Qwen Image Edit](../libraries/qwen_image/README.md) |
| Qwen Image 2.1 | Porter le débruiteur, le VAE et le conditionnement Qwen 3.5 9B. Vérifier les formats INT8 ConvRot et FP8 mixtes. |
| Qwen3-VL, Qwen 3.5 9B | Ajouter les profils multimodaux ou quantifiés utilisés par les nouveaux workflows. Les formats repérés incluent INT8 ConvRot, FP8 à échelles, NVFP4 AWQ et GPTQ W4A8. |
| MiniMax H3 et ses dérivés | Porter le débruiteur et les VAE image, vidéo et audio. Réutiliser le portage Qwen3-VL pour le conditionnement après validation des formats H3. |
| MOSS-Image et Krea 2 | Porter leurs graphes de diffusion et valider les formats INT8 ConvRot et FP8 à échelles respectifs. |
| VAE FLUX.2 | Porter le décodeur du profil `flux2-vae_small` à 32 canaux latents. |
| LoRA Qwen Image, MiniMax H3 et Krea 2 | Ajouter les conventions de clés de chaque famille après le portage du débruiteur correspondant. |

FLUX.1 Canny Dev réutilise les trois composants testés ci-dessous. MiniMax H3 utilise son profil Qwen3-VL pour le conditionnement.

## Tests

```powershell
$env:ZML_TEMP_ROOT = 'D:\AI\zml_lib_Temp'
& $py -B -m unittest discover -s tests -p test_model_catalog.py -v
```

Les tests vérifient les trois profils RRDBNet, l’ordre de sélection des poids et le rejet d’un cache altéré. Ils exercent les deux signatures SD 1.5 et le choix du profil à transmettre au convertisseur avec des checkpoints synthétiques et une doublure du convertisseur. Ils couvrent aussi les signatures CLIP-L, T5-XXL et VAE FLUX.1, ainsi que les états du scan pour des fichiers inconnus, illisibles et partiels. Les fichiers d’essai sous `ZML_TEMP_ROOT` sont retirés à la fin.

Les trois composants suivants ont été préparés depuis `ComfyUI-Shared/models` et exécutés sous Windows x64 sur la RTX 3090 via `iree_cuda`. Les trois préparations ont été reconnues par le catalogue et le second appel a réutilisé le cache. T5-XXL a effectué deux inférences avec le modèle chargé.

| Fichier local | Poids préparés | Sortie comparée à PyTorch | Écart absolu maximal |
| --- | --- | --- | ---: |
| `text_encoders/clip_l.safetensors` | 196 tenseurs FP32 | 4 tokens, `[1, 4, 768]` | `5,25e-6` |
| `text_encoders/t5xxl_fp16.safetensors` | 219 tenseurs FP16 | 9 tokens, `[1, 9, 4096]` | `8,58e-6` |
| `vae/ae.safetensors` | 138 tenseurs FP32 | RGB FP32, `[1, 3, 32, 48]` | `4,05e-6` |

Pour comparer les nouveaux composants à PyTorch sur la RTX 3090, configurer le plugin CUDA et le compilateur IREE, puis lancer la commande suivante. Chaque composant est préparé dans son propre dossier de test, vérifié après réutilisation du cache et comparé à la référence CPU avec les poids d’origine. Le script contrôle le placement de la sortie sur CUDA avant sa copie vers l’hôte. [Configuration CUDA sous Windows](../libraries/botsort/README.md#validation-cuda-sous-windows-x64)

```powershell
$shared = 'D:\AI\Comfy-Desktop\ComfyUI-Shared\models'
$cuda = @('--@zml//platforms:cuda=true', '--@zml//platforms:rocm=false', '--@zml//platforms:vulkan=false')
& $py -B .\scripts\build.py --target //libraries/clip:clip_text_probe --target //libraries/t5:t5_encoder `
  --target //libraries/flux1:flux1_probe -- --repo_contents_cache= --jobs=16 --host_compilation_mode=fastbuild @cuda `
  --then $py -B .\tests\windows\validate_catalog_component.py --bin-dir '{bin}' `
  "$shared\text_encoders\clip_l.safetensors" "$shared\text_encoders\t5xxl_fp16.safetensors" "$shared\vae\ae.safetensors"
```

Sous Windows, le catalogue a reconnu et préparé trois checkpoints locaux de ComfyUI, `RealESRGAN_x4plus`, `4x_RealisticRescaler_100000_G` et `4x_NMKD-Siax_200k`. Le second appel a réutilisé le cache pour les trois. Avec les poids x4plus préparés par le catalogue, la comparaison sur CPU avec PyTorch via Spandrel donne un écart maximal de `2,00e-6` en sortie FP32 et un PNG identique. Le CLI et le cycle de vie de l’ABI C passent sur CPU. Le CLI CUDA archivé s’arrête sur `CUDA_ERROR_INVALID_CONTEXT` avec les poids produits par les deux chemins de conversion.
