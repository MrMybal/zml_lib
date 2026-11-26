# Qwen2.5-VL 7B

Cette bibliothèque produit les états cachés FP32 par token pour le conditionnement de Qwen Image et Qwen Image Edit 2511. Le ViT visuel ZML reprend la projection des patches RGB, les 32 blocs d’attention et la fusion spatiale 2 × 2. Les blocs 7, 15, 23 et 31 regardent tous les patches du même cadre. Les autres restreignent l’attention à leur fenêtre. Le décodeur texte réutilise les couches Qwen du dépôt, ajoute les positions MRoPE temps, hauteur et largeur et accepte les vecteurs projetés du ViT à la place des tokens image. [Configuration Qwen2.5-VL 7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/config.json) · [Workflow ComfyUI Qwen Image Edit 2511](https://github.com/Comfy-Org/docs/blob/main/tutorials/image/qwen/qwen-image-edit-2511.mdx)

Le convertisseur lit le fichier local `qwen_2.5_vl_7b_fp8_scaled.safetensors`. Il extrait 338 tenseurs de texte et 390 tenseurs de vision, applique les échelles de poids aux matrices E4M3FN et écrit les deux composants en FP16. ZML conserve les poids en FP16 sur le périphérique et calcule en FP32. Le checkpoint peut contenir la projection vocabulaire pour le dialogue. Ce port produit les embeddings de conditionnement. Le prétraitement des images, la tokenisation et la sélection des tokens à passer à la diffusion sont à la charge de l’appelant.

## Format du lanceur

`qwen25_vl_encode.exe PROFIL BACKEND POIDS N ENTREE SORTIE [ENTREE SORTIE ...]` traite une ou plusieurs entrées de même taille avec le même modèle chargé. Il vérifie le périphérique de chaque sortie avant sa copie vers l’hôte. L’appelant possède les fichiers. L’entrée et la sortie sont des tableaux FP32 little-endian contigus.

| Profil | Entrée | Sortie |
| --- | --- | --- |
| `vision` | `[N,1180]`. Chaque ligne contient les indices de ligne et de colonne, le numéro de fenêtre, le numéro de cadre et les 1176 valeurs du patch `[3,2,14,14]`. Grouper chaque carré 2 × 2 de patches dans l’ordre haut gauche, haut droit, bas gauche, bas droit. `N` est un multiple de 4 entre 4 et 2048. | `[1,N/4,3584]` dans l’ordre d’entrée des groupes de patches. |
| `qwen25_vl_7b` | `[N,3589]`. Chaque ligne contient l’identifiant du token, les positions temps, ligne, colonne, un masque binaire, puis 3584 valeurs. Utiliser l’identifiant `-1` pour injecter le vecteur de la ligne. Les trois positions d’un token de texte sont égales. `N` est compris entre 1 et 512. | `[1,N,3584]`. Le masque d’entrée permet à l’appelant d’écarter les lignes de remplissage. |

`inputs.py` forme ces tableaux avec `vision_records`, `text_records` et `image_positions`. Pour les pixels, `Qwen2VLImageProcessor` de Transformers fournit les patches normalisés et la grille `image_grid_thw`. Donner la grille à `vision_records` pour obtenir les identifiants des fenêtres. Après l’encodage, remplacer chaque emplacement image du prompt par un vecteur de sortie du ViT. L’utilitaire `image_positions` fournit les positions MRoPE du cas à une image. Le lanceur expose les deux schémas avec `vision schema` et `qwen25_vl_7b schema`.

## Validation Windows x64

Configurer le plugin et le compilateur IREE pour CUDA. Le script prépare les poids à partir du fichier ComfyUI, compare les deux encodeurs et le trajet image vers conditionnement à Transformers FP32 avec les mêmes valeurs de poids, puis supprime ses fichiers d’essai. Omettre `--source` et `--tokenizer` pour rejouer le test réduit de MRoPE et du masque. [Configuration CUDA](../botsort/README.md#validation-cuda-sous-windows-x64)

```powershell
$env:ZML_TEMP_ROOT = 'D:\AI\zml_lib_Temp'
$py = 'D:\AI\ComfyUI\.venv\Scripts\python.exe'
$source = 'D:\AI\Comfy-Desktop\ComfyUI-Shared\models\text_encoders\qwen_2.5_vl_7b_fp8_scaled.safetensors'
$tokenizer = 'D:\AI\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\text_encoders\qwen25_tokenizer'
$flags = @('--@zml//platforms:cuda=true', '--@zml//platforms:rocm=false', '--@zml//platforms:vulkan=false')
& $py -B .\scripts\build.py --target //libraries/qwen25_vl:qwen25_vl_encode `
  -- --repo_contents_cache= --jobs=16 --host_compilation_mode=fastbuild @flags `
  --then $py -B .\libraries\qwen25_vl\validate.py --probe '{bin}\libraries\qwen25_vl\qwen25_vl_encode.exe' `
  --source $source --tokenizer $tokenizer --expected-device 'RTX 3090'
```

Pour produire les deux fichiers réutilisables séparément, appeler `prepare.py --component text --source $source --probe $probe --output $textWeights` et `prepare.py --component vision --source $source --probe $probe --output $visionWeights`. Les fichiers de destination doivent être nouveaux, dans un dossier existant.

### Résultats

Le checkpoint ComfyUI local a pour SHA-256 `cb5636d852a0ea6a9075ab1bef496c0db7aef13c02350571e388aea959c5c0b4`. La conversion a appliqué les échelles de 196 matrices du décodeur texte et 162 du ViT. Chaque encodeur a traité deux entrées successives avec un seul chargement. Les quatre sorties du modèle complet ont été vérifiées sur la RTX 3090 via `iree_cuda`, sous Windows x64.

| Comparaison à Transformers FP32 | Tokens comparés | Écart absolu maximal | RMSE |
| --- | ---: | ---: | ---: |
| ViT sur deux images de 196 × 140, quatre fenêtres chacune | 35 par image | `7,35e-4` | `3,84e-5` |
| Texte avec masque de remplissage | 9 | `2,32e-3` | `4,47e-5` |
| Décodeur multimodal avec les mêmes embeddings visuels | 63 | `7,10e-3` | `8,26e-5` |
| Chaîne image vers conditionnement avec les deux ViT | 63 | `7,76e-3` | `1,16e-4` |

Le test du décodeur 7B utilise une tolérance absolue de `4e-3`, une tolérance relative de `2e-3` et une RMSE maximale de `2,5e-4`. Il compare les lignes visibles du texte. Sur les 54 lignes de remplissage écartées du conditionnement, l’écart maximal brut était de `4,13e-2`. Le test réduit a couvert la position multimodale, le remplissage et le rejet d’une position invalide. Les régressions Qwen2 et Qwen3 ont retrouvé les séquences PyTorch sur CPU Windows.
