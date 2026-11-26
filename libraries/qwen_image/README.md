# Qwen Image Edit 2511

Le débruiteur ZML calcule les 60 blocs à deux flux du checkpoint 2511. Il applique RoPE aux coordonnées des patches et aux tokens texte. Il conditionne l’image cible sur le pas courant et les images de référence sur le pas zéro. Les références participent à l’attention. La sortie contient les vitesses des patches de l’image cible. Le texte peut provenir du [port Qwen2.5‑VL 7B](../qwen25_vl/README.md). [Modèle Qwen Image Edit 2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) · [Pipeline de référence](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_plus.py)

Le convertisseur conserve les codes et les échelles des 839 matrices FP8 E4M3FN du fichier ComfyUI. Les tenseurs ordinaires sont stockés en FP32. Le graphe décode les valeurs de poids en FP32 avant les calculs. Le VAE traite une image fixe avec 16 canaux latents et un facteur spatial de 8. Pour une seule image, les convolutions causales utilisent leur dernier coefficient temporel et les deux étages temporels de chaque moitié sont contournés. Les sorties de l’encodeur sont les moyennes normalisées de la distribution latente.

L’intégration fournit la bibliothèque Zig et une sonde sur tenseurs. Le pipeline de validation compose les encodeurs, le VAE et le débruiteur. Il couvre les résolutions 512 × 512 et 1024 × 1024 sur Colab L4.

## Fichiers et lanceur

`prepare.py --probe $probe --source $checkpoint --output $weights --profile PROFIL` vérifie les noms et dimensions attendus par la sonde. Utiliser le fichier `qwen_image_vae.safetensors` pour les profils `vae_encode` et `vae_decode`, puis `qwen_image_edit_2511_fp8mixed.safetensors` pour `edit_2511`. La destination doit être un nouveau fichier dans un dossier existant. Le modèle 2511 préparé occupe environ 20,74 Go. [Poids ComfyUI](https://github.com/Comfy-Org/docs/blob/main/tutorials/image/qwen/qwen-image-edit-2511.mdx)

```text
qwen_image_probe.exe vae_encode BACKEND POIDS H W ENTREE SORTIE
qwen_image_probe.exe vae_decode BACKEND POIDS H W ENTREE SORTIE
qwen_image_probe.exe edit_2511 BACKEND POIDS CIBLES PATCHES TOKENS PAS ENTREE SORTIE
```

Les fichiers de données sont contigus en FP32 little-endian. L’appelant possède les fichiers et choisit le backend. Le VAE échange des données planaires entre `[1,3,8H,8W]` dans `[-1,1]` et `[1,16,H,W]` normalisées.

Pour le débruiteur, `PATCHES` comprend les patches cibles en premier, puis ceux des références. Le plafond est de 4 096 patches en mode dense et de 8 192 avec le mode CUDA Linux par blocs et le module de référence. Les deux modes acceptent jusqu’à 512 tokens texte. L’entrée contient successivement `[PATCHES,64]` de latents, `[PATCHES,3]` de positions, `[TOKENS,3584]` d’embeddings, `[TOKENS,3]` de positions texte, un masque binaire `[TOKENS]` et `PAS+1` temps normalisés décroissants. Avec `PAS=0`, la sonde retourne la vitesse à l’instant fourni. Avec `PAS>0`, elle exécute Euler avec le modèle chargé une fois et retourne les latents finaux `[CIBLES,64]`. Le programme vérifie le placement de chaque sortie avant sa copie vers l’hôte. `inputs.py` sérialise ce format.

Pour examiner une trajectoire de référence avec `PAS>0`, ajouter `--reference-trajectory` après `SORTIE` et ajouter en fin d’entrée les `PAS-1` états cibles de référence après le premier pas, chacun en `[CIBLES,64]`. La sortie commence par les latents finaux de l’intégration ZML habituelle, puis contient une vitesse `[CIBLES,64]` par pas sur les états de référence. Le premier pas partage le même état initial. La sonde réutilise le modèle et le graphe compilé pour toutes les prédictions.

Avec deux pas, `--reference-feedback` accepte la même entrée et écrit onze lignes `[CIBLES,64]`. Les cinq premières contiennent le latent final ZML, la première vitesse, le latent réellement réinjecté, la vitesse du second pas sur ce latent et la vitesse sur le latent de référence. Les six suivantes contiennent trois paires état et vitesse à 25 %, 50 % et 75 % entre le latent de référence et le latent ZML.

## Éditer une image avec le lanceur direct

`generate.py` compose Qwen2.5-VL, le VAE et le débruiteur à partir des trois checkpoints ComfyUI dans `--model-root`. Le texte et l’image conditionnent une trajectoire Euler. `--shift` règle le décalage du calendrier de bruit et vaut 1,15 par défaut. Le lanceur redimensionne l’image de référence à la résolution de sortie avant le VAE. `--vision-size` règle la taille de l’image passée à l’encodeur visuel.

```bash
python -B libraries/qwen_image/generate.py \
  --probe "$qwen_probe" --encoder-probe "$encoder_probe" \
  --model-root "$models" --tokenizer "$comfy/comfy/text_encoders/qwen25_tokenizer" \
  --image reference.png --prompt "Replace the orange shape with a purple square." \
  --output edited.png --width 512 --height 512 --steps 20 --seed 42 \
  --workspace "$scratch" --cuda-low-memory --cuda-math-library "$cuda_math" \
  --expected-device cuda:0 --expected-pjrt cuda
```

Les variables désignent des chemins locaux. Sur Colab, `$cuda_math` désigne le module compilé avec les bibliothèques CUDA des runfiles de la sonde. Le lanceur accepte aussi `--prompt-file` pour le texte UTF-8. Il crée un PNG RGB à un nouveau chemin, émet un objet JSON sur stdout et écrit les diagnostics sur stderr. Le répertoire de travail créé pour l’appel est supprimé à la sortie.

Sur Colab L4, un essai avec `--steps 2 --seed 42` et les poids ComfyUI a produit un PNG RGB de 512 × 512 avec cette instruction. Le JSON confirme `cuda:0` pour le débruiteur et le VAE. Le test a vérifié l’intégrité du PNG et la présence de valeurs distinctes dans ses canaux. Le raccordement du lanceur à la CLI ImageStudio reste à faire.

## Validation Windows x64

La commande utilise l’installation locale ComfyUI pour la référence PyTorch. Elle décompose les poids, compose le conditionnement avec Qwen2.5-VL, exécute le débruiteur puis le VAE et retire les fichiers temporaires. Le test utilise une image de 64 × 48 et deux pas d’Euler conditionnels. La référence du débruiteur s’exécute sur la RTX 3090 et celle du VAE sur CPU. Le test réduit couvre séparément une prédiction et deux pas avec deux références. [Configuration CUDA](../botsort/README.md#validation-cuda-sous-windows-x64)

```powershell
$env:ZML_TEMP_ROOT = 'D:\AI\zml_lib_Temp'
$env:NVIDIA_TF32_OVERRIDE = '0'
$py = 'D:\AI\ComfyUI\.venv\Scripts\python.exe'
$models = 'D:\AI\Comfy-Desktop\ComfyUI-Shared\models'
$comfy = 'D:\AI\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI'
$flags = @('--@zml//platforms:cuda=true', '--@zml//platforms:rocm=false', '--@zml//platforms:vulkan=false')
& $py -B .\scripts\build.py --target //libraries/qwen_image:qwen_image_probe --target //libraries/qwen25_vl:qwen25_vl_encode `
  -- --repo_contents_cache= --jobs=16 --host_compilation_mode=fastbuild @flags `
  --then $py -B .\libraries\qwen_image\validate.py --probe '{bin}\libraries\qwen_image\qwen_image_probe.exe' `
  --comfy $comfy --backend cuda --expected-device 'RTX 3090' --reference-device cuda:0 --vae "$models\vae\qwen_image_vae.safetensors" `
  --denoiser "$models\diffusion_models\qwen_image_edit_2511_fp8mixed.safetensors" `
  --encoder "$models\text_encoders\qwen_2.5_vl_7b_fp8_scaled.safetensors" `
  --encoder-probe '{bin}\libraries\qwen25_vl\qwen25_vl_encode.exe' --tokenizer "$comfy\comfy\text_encoders\qwen25_tokenizer"
```

Omettre les chemins des checkpoints pour exécuter le test réduit, qui couvre une vitesse avec des poids FP32 et deux pas avec des poids FP8 à échelles. Fournir seulement `--vae` pour comparer les deux moitiés du VAE. Pour comparer les deux références PyTorch FP32 après une seule exécution ZML, ajouter `--reference-device cpu` à la commande. Le script vérifie les latents et le RGB après le décodage indépendant de chaque résultat et mesure aussi l’écart entre les deux références.

Ajouter `--step-diagnostics` à la commande complète pour comparer les vitesses aux pas `0,78` et `0,35` avec les mêmes entrées pour ZML et PyTorch. Le script prend la première référence demandée pour les états intermédiaires. Il réintègre ensuite les vitesses sur ces états et mesure l’effet de réinjecter les latents ZML à chaque pas. Il affiche aussi l’erreur de l’intégration hôte avec les vitesses PyTorch et conserve les critères de validation du résultat final.

Ajouter `--feedback-diagnostics` pour comparer le deuxième pas sur les deux latents intermédiaires et sur trois interpolations. Chaque comparaison ZML et PyTorch utilise le même latent sérialisé. Le script vérifie les mises à jour d’Euler et mesure le changement de sortie de PyTorch quand on lui donne le latent ZML. La validation finale porte sur la trajectoire indépendante de chaque moteur.

Ajouter `--block-diagnostics` à la commande complète pour comparer les 60 blocs du premier pas, ainsi que les projections d’entrée et la sortie du débruiteur. Le script mesure d’abord le chemin enchaîné, puis réexécute chaque bloc avec les états image et texte et le conditionnement temporel de PyTorch. La RMSE relative est la RMSE divisée par la racine de la moyenne des carrés de la référence. Le script compare la vitesse du graphe instrumenté à celle du graphe standard et applique les seuils habituels à la sortie du graphe standard.

Pour reproduire la sonde seule, utiliser `PAS=0` et ajouter `--reference-blocks` après `SORTIE`. Après l’entrée ordinaire, ajouter une valeur de contrôle réservée à la sonde, le conditionnement PyTorch après SiLU `[2,3072]`, puis 61 paires d’états `[PATCHES,3072]` et `[TOKENS,3072]`. La première paire suit les projections d’entrée. Chaque autre paire suit un bloc. Les deux fichiers `SORTIE` et `SORTIE.forced` contiennent la vitesse `[CIBLES,64]`, le conditionnement ZML et les 61 paires d’états calculées. Le script de validation prépare ces données et retire les fichiers de mesure.

`--reference-prefix=1` accepte cette même entrée, remplace la sortie du premier bloc par la paire d’états PyTorch et poursuit avec le conditionnement temporel ZML. La sonde écrit uniquement la vitesse.

Le graphe exprime les phases de RoPE en FP64. Le compilateur IREE de cette installation abaisse ces opérations en FP32 par défaut. L’option IREE `--iree-input-demote-f64-to-f32=false` permet de conserver les types FP64 du graphe pour un essai de précision.

Mesures obtenues avec `iree_cuda` sur la NVIDIA GeForce RTX 3090, `zml_win` au commit `ed3be5bffb1e84cdb9daa50b36959de996dfdeff` et ComfyUI au commit `2fcb6cef9ab1b6dc556b7cff1765a0724ad5e884`.

| Test | Écart absolu maximal | RMSE |
| --- | ---: | ---: |
| Débruiteur réduit, prédiction FP32 | `9,54e-7` | `3,09e-7` |
| Débruiteur réduit, deux pas FP8 à échelles avec deux références | `5,96e-7` | `1,77e-7` |
| Débruiteur réduit FP8, deuxième vitesse sur état imposé | `9,54e-7` | `2,69e-7` |
| VAE réel, encodage normalisé | `2,98e-6` | `5,32e-7` |
| VAE réel, décodage RGB | `4,17e-6` | `3,55e-7` |

Avec les poids réels et le conditionnement Qwen2.5-VL, la chaîne a produit un PNG de 64 × 48 après deux pas d’Euler sur la RTX 3090. La comparaison numérique complète dépasse les seuils du test. La limite de RMSE est `1e-3` et les tolérances absolues à l’élément sont `4e-3` pour les latents et `5e-3` pour le RGB, avec une tolérance relative de `3e-3`.

| Référence PyTorch FP32 | Écart maximal des latents | RMSE des latents | Écart maximal RGB dans `[-1,1]` | RMSE RGB |
| --- | ---: | ---: | ---: | ---: |
| CPU | `2,41e-2` | `6,20e-3` | `1,39e-1` | `1,42e-2` |
| CUDA | `7,96e-3` | `2,33e-3` | `3,44e-2` | `3,49e-3` |

Les références PyTorch CPU et CUDA diffèrent elles-mêmes de `2,84e-2` au maximum sur les latents du même essai. Le PNG ZML correspondant au tableau avait le SHA-256 `6d80bf880b383596df3d45bd7c0c9013b871dceecd329c9c9d1c52c14a09056e`.

L’ordre de multiplication du facteur d’attention a été vérifié sur six blocs du checkpoint dans PyTorch CUDA avec une entrée synthétique. Appliquer `1/√d` à `Q` avant le produit déplace les latents de `1,11e-2` au maximum après deux pas. L’appliquer après le produit reproduit la référence ComfyUI sur ce test ciblé. Dans ZML avec les 60 blocs, cette variante porte l’écart maximal face à PyTorch CUDA à `3,21e-2` sur les latents et à `1,69e-1` sur le RGB. Le port conserve donc la multiplication avant le produit, qui donne les valeurs du tableau ci-dessus avec les réglages IREE par défaut. En conservant FP64 pour RoPE sur les mêmes entrées, les écarts maximaux face à PyTorch CUDA sont de `7,95e-3` sur les latents et de `3,35e-2` sur le RGB. Le test complet dépasse toujours ses seuils.

Le même essai 64 × 48, relancé avec `--step-diagnostics` et PyTorch sur la RTX 3090, produit le PNG du tableau avec le même SHA-256.

| Comparaison à chaque pas ou après intégration | Écart maximal | RMSE |
| --- | ---: | ---: |
| Vitesse au pas `0,78` avec le même latent initial | `6,79e-3` | `1,61e-3` |
| Vitesse au pas `0,35` avec le latent PyTorch | `8,32e-4` | `2,36e-4` |
| Latents finaux en intégrant les vitesses PyTorch sur l’hôte | `0` | `0` |
| Latents finaux en intégrant les vitesses ZML évaluées sur les latents PyTorch | `3,08e-3` | `6,97e-4` |
| Effet de la réinjection du latent ZML au deuxième pas | `8,08e-3` | `2,22e-3` |

L’écart apparaît dans la première prédiction du débruiteur. La réinjection du premier latent ZML lors du deuxième pas augmente l’écart final. L’intégration hôte avec les vitesses PyTorch reproduit exactement les latents de référence sur ce test. La comparaison complète reste au-delà des seuils sur les latents finaux et le RGB.

Injecter le conditionnement temporel calculé par PyTorch CUDA pour les pas `0,78` et zéro fait passer l’écart maximal de la première vitesse de `6,79e-3` à `6,60e-3`. Les autres entrées viennent du même essai. Dans un test isolé du conditionnement, l’écart maximal entre PyTorch CPU et CUDA est de `4,58e-5` au pas `0,78`.

IREE accepte `--iree-input-demote-f64-to-f32=false` dans l’API de session utilisée par PJRT. Sur CUDA, une réduction FP64 minimale compile avec `--iree-codegen-llvmgpu-use-reduction-vector-distribution=false` dans `iree-compile`. L’API de session rejette cette option ainsi que `--iree-dispatch-creation-split-matmul-reduction=4`. Avec la distribution vectorielle par défaut et la démotion FP64 désactivée, la compilation du modèle complet avec les normalisations FP64 échoue sur la réduction de largeur 3 584.

Le fractionnement explicite en quatre des projections FP8 s’exécute sur la RTX 3090 sans option de compilation supplémentaire. L’écart maximal de la première vitesse descend à `6,47e-3`. La RMSE finale monte à `2,42e-3` sur les latents et `3,62e-3` sur le RGB, contre `2,33e-3` et `3,49e-3` avec le calcul standard. Le débruiteur garde ce calcul standard. Les seuils de comparaison restent dépassés.

Les sommes par paires remplacent la réduction FP64 par des tranches et des additions successives. Cette variante compile et s’exécute sur les 60 blocs avec `--iree-input-demote-f64-to-f32=false`. Le tableau compare les mêmes latents, le même conditionnement Qwen2.5-VL et la même référence PyTorch CUDA. RoPE reste en FP64 dans chaque essai.

| Calcul | RMSE première vitesse | RMSE finale des latents | RMSE finale RGB |
| --- | ---: | ---: | ---: |
| Témoin avec RoPE FP64 | `1,61e-3` | `2,315e-3` | `3,482e-3` |
| LayerNorm et RMSNorm FP64 par paires | `1,95e-3` | `5,555e-3` | `9,906e-3` |
| Exponentielle et somme du softmax FP64 par paires | `1,61e-3` | `2,317e-3` | `3,479e-3` |
| Normalisations et softmax FP64 par paires | `1,95e-3` | `5,584e-3` | `9,960e-3` |
| Produits et sommes des modulations en FP64, arrondis en FP32 | `1,91e-3` | `4,730e-3` | `8,102e-3` |

Le dernier essai élargit le produit et la somme des opérations de modulation qui correspondent à `addcmul` dans PyTorch, puis arrondit le résultat en FP32. Les trois variantes de réduction et cet essai dépassent les seuils du test final. Le débruiteur conserve son calcul standard.

### Diagnostic CUDA des blocs du premier pas

Le test des 60 blocs utilise les mêmes 12 patches cibles, 12 patches de référence, 22 tokens et le même temps `0,78` que le test complet. Sur la RTX 3090, la première vitesse du graphe instrumenté est identique à celle du graphe standard. Alimenter la tête de sortie ZML avec le dernier état PyTorch et son conditionnement donne une erreur maximale de vitesse de `5,25e-6`. L’erreur maximale est de `6,79e-3` sur le chemin enchaîné.

| Mesure à partir des états PyTorch | RMSE relative image | RMSE relative texte |
| --- | ---: | ---: |
| Bloc 1 | `1,05e-4` | `2,51e-6` |
| Maximum sur les blocs 2 à 60 | `3,83e-6` (bloc 59) | `1,18e-6` (bloc 34) |

Dans le bloc 1 rejoué isolément, la RMSE relative du flux image passe de `8,45e-7` à l’entrée de l’attention à `2,76e-4` à sa sortie. Les scores PyTorch montent à `37 397,44`. À cette échelle, l’arrondi FP32 influence la distribution du softmax lorsque deux scores sont proches.

Sur PyTorch CUDA, réinjecter seulement les requêtes ZML donne une RMSE relative de `1,70e-4` en sortie d’attention. Réinjecter seulement les clés donne `1,53e-4` et réinjecter seulement les valeurs donne `1,33e-6`. Réinjecter les scores ZML avec les valeurs PyTorch reproduit la RMSE relative de `2,76e-4`. Ces essais situent l’amplification dans les scores de l’attention du premier bloc, avec une contribution des arrondis des requêtes et des clés.

Les variantes suivantes modifient seulement le premier bloc et utilisent la même référence PyTorch CUDA. Les deux essais FP64 conservent aussi RoPE en FP64. Le produit matriciel QK FP64 direct échoue à la compilation IREE lors de la distribution vectorielle. L’essai correspondant emploie un produit suivi de sommes par paires.

| Calcul | RMSE relative image du bloc 1 isolé | RMSE finale latents | RMSE finale RGB |
| --- | ---: | ---: | ---: |
| Standard | `1,05e-4` | `2,33e-3` | `3,49e-3` |
| Échelle d’attention après le produit scalaire | `7,91e-5` | `8,21e-3` | `1,60e-2` |
| Scores QK sommés par paires en FP64 | `1,02e-4` | `7,53e-3` | `1,44e-2` |
| Projections Q et K par groupes de 256 et moyenne des carrés en FP64 | `9,85e-5` | `2,80e-3` | `4,51e-3` |

La parité de la chaîne complète reste hors seuil sur ce cas. Le débruiteur conserve le calcul standard et le diagnostic peut être rejoué avec `--block-diagnostics`.

### Projection Q et K compatible avec la référence CUDA

Le profil `edit_2511_cuda_qk` utilise le même fichier de poids et le même format d’appel que `edit_2511`. Pour le premier bloc, il accumule les projections Q et K en trois partitions de largeur 1 024. Il suit l’ordre d’accumulation FP32 de cuBLAS mesuré sur la RTX 3090 pour 24 patches image et 22 tokens texte. La RMSNorm de largeur 128 reproduit l’accumulation par groupes de quatre valeurs du noyau CUDA PyTorch. Les produits-sommes sont exprimés en FP64, puis chaque somme est arrondie en FP32 avant la suivante. [Noyau CUDA PyTorch 2.10](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cuda/layer_norm_kernel.cu#L172)

Dans la commande de validation Windows ci-dessus, utiliser ces options du compilateur IREE et ajouter `--cuda-qk --qk-diagnostics` aux arguments de `validate.py`.

```powershell
$env:IREE_PJRT_IREE_COMPILER_OPTIONS = '--iree-cuda-target=sm_86 --iree-input-demote-f64-to-f32=false'
```

Le diagnostic donne à chaque opérateur les activations de PyTorch CUDA. Avec les poids ComfyUI, les quatre projections Q et K des flux image et texte, leurs RMSNorm et leurs rotations RoPE ont donné un écart nul aux douze comparaisons du premier pas. Dans le chemin complet, ces opérateurs reçoivent les activations produites par ZML à l’étape précédente.

| Calcul en 64 × 48 et deux pas | RMSE première vitesse | RMSE finale latents | RMSE finale RGB |
| --- | ---: | ---: | ---: |
| Profil standard | `1,61e-3` | `2,33e-3` | `3,49e-3` |
| Profil CUDA Q/K | `1,42e-3` | `2,90e-3` | `4,69e-3` |
| CUDA Q/K, LayerNorm à l’entrée de l’attention et échelle après le produit QK | `1,03e-3` | `3,51e-3` | `5,84e-3` |
| Même essai avec accumulation explicite du produit QK | `1,04e-3` | `3,48e-3` | `5,79e-3` |

Avec le profil CUDA Q/K, intégrer les deux vitesses ZML obtenues sur les latents PyTorch donne une RMSE de `6,10e-4` sur le latent final. La réinjection du latent ZML au deuxième pas ajoute un écart de `2,86e-3` de RMSE. La correction locale est proposée par le profil CUDA optionnel. La parité après deux pas indépendants reste hors seuil.

### Rejeu croisé du deuxième pas

`--cuda-qk --feedback-diagnostics --reference-device cuda:0` a été vérifié sous Windows x64 sur la RTX 3090, avec les poids ComfyUI et le conditionnement Qwen2.5-VL du test 64 × 48. Les deux mises à jour d’Euler reproduisent exactement le calcul hôte. Les cinq cas suivants utilisent le même temps `0,35` et les mêmes entrées pour les deux moteurs.

| Fraction du déplacement du latent ZML | RMSE du latent intermédiaire | RMSE du changement de vitesse dans PyTorch | RMSE des vitesses ZML et PyTorch sur le même latent | RMSE du latent final avec la deuxième vitesse PyTorch |
| --- | ---: | ---: | ---: | ---: |
| 0 % | `0` | `0` | `1,35e-4` | `0` |
| 25 % | `1,53e-4` | `1,93e-3` | `1,98e-4` | `6,86e-4` |
| 50 % | `3,06e-4` | `3,91e-3` | `1,40e-4` | `1,39e-3` |
| 75 % | `4,59e-4` | `5,93e-3` | `1,69e-4` | `2,10e-3` |
| 100 % | `6,12e-4` | `8,20e-3` | `1,59e-4` | `2,896e-3` |

Sur le latent ZML, la deuxième prédiction PyTorch reproduit presque toute la dérive finale de ZML, mesurée à `2,901e-3` sur les latents. La variation de vitesse est environ 13,4 fois plus grande que le déplacement du latent entre les deux moteurs à l’entrée du pas.

Au premier pas, un essai de localisation remplace les états image et texte produits par le bloc 1 par les états de PyTorch avant de poursuivre le débruitage ZML. Cette substitution réduit la RMSE de la première vitesse de `1,42e-3` à `3,95e-5`. Elle situe l’écart à corriger avant l’entrée du bloc 2.

| Variante testée avec les mêmes entrées | RMSE de la première vitesse | RMSE finale latents | RMSE finale RGB |
| --- | ---: | ---: | ---: |
| Profil CUDA Q/K | `1,42e-3` | `2,90e-3` | `4,69e-3` |
| Centrage en FP64 des scores du premier bloc avant arrondi FP32 | `2,24e-3` | `8,10e-3` | `1,61e-2` |
| Alignement des projections d’entrée, du temps et des normalisations | `1,79e-3` | `2,99e-3` | `5,31e-3` |
| Alignement des entrées et centrage des scores | `2,60e-3` | `8,24e-3` | `1,70e-2` |

| Checkpoint ComfyUI | SHA-256 |
| --- | --- |
| `qwen_image_edit_2511_fp8mixed.safetensors` | `c9fdc158e46d3b61ef75f21ae866ca2fe808bf4a53643120d1c1e87c19280a4e` |
| `qwen_image_vae.safetensors` | `a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f` |

## Validation sur une L4 sous Linux x64

`validate_linux.py` compare les deux pas d’Euler, le latent final et le RGB avec les trois checkpoints ComfyUI. Le débruiteur et le VAE de référence s’exécutent dans PyTorch CUDA. Le lanceur sélectionne une L4 par UUID pour les processus de test, exige le placement ZML sur `cuda:0` avec PJRT `cuda` et affiche la VRAM libre avant de préparer les poids. Le fichier du débruiteur préparé prend environ 19,31 Gio. Le chargement et les calculs demandent de la mémoire supplémentaire.

Le dossier `zml_win` doit être à côté de `zml_lib` au commit de `zml.lock.json`. Pour reprendre la référence Windows, utiliser le commit ComfyUI `2fcb6cef9ab1b6dc556b7cff1765a0724ad5e884` et PyTorch `2.10.0+cu128`. Le dossier des modèles garde les sous-dossiers `diffusion_models`, `vae` et `text_encoders`. Les fichiers de test et de compilation résident dans un sous-dossier éphémère du disque local de la VM. Le lanceur le supprime en fin de commande. [Compilation CUDA avec ZML](https://docs.zml.ai/howtos/deploy_on_server/) · [Versions de PyTorch](https://pytorch.org/get-started/previous-versions/)

```bash
python -B libraries/qwen_image/validate_linux.py \
  --bazel /content/tools/bazel-9.3.0 \
  --scratch /content/zml-scratch --comfy /content/ComfyUI \
  --models /content/models --cuda-low-memory --cuda-reference-math \
  --width 512 --height 512 --vision-size 224
```

Avec des sondes Linux déjà compilées, passer `--probe` et `--encoder-probe` avec leurs runfiles. `--cuda-qk` sélectionne le profil utilisé pour le rejeu Windows. Ce profil suit l’ordre d’accumulation observé sur la RTX 3090. La comparaison sur L4 utilise ses propres mesures d’erreur et conserve les seuils d’origine.

`--cuda-reference-math` compile le module compagnon CUDA pour la L4 avec `nvcc` et les bibliothèques cuBLAS 13 du moteur ZML. Passer `--nvcc` pour sélectionner le compilateur. Le lanceur transmet le chemin de la bibliothèque à la sonde via `ZML_QWEN_CUDA_MATH` et la supprime avec le dossier temporaire. Pour dépasser 4 096 jetons image, le débruiteur utilise ce mode avec `--cuda-low-memory`. L’attention traite jusqu’à 1 024 requêtes à la fois et réutilise l’espace de calcul des scores. Chaque requête conserve toutes les clés et valeurs. Ce mode accepte jusqu’à 8 192 jetons image et 512 jetons texte, selon la VRAM disponible. Sans options de taille, le lanceur reprend le cas 64 × 48.

Le lanceur applique les options XLA de parité au seul débruiteur. L’encodeur Qwen2.5-VL conserve les réglages CUDA par défaut, qui réduisent la mémoire nécessaire à la projection des séquences longues. `ZML_QWEN_DENOISER_XLA_FLAGS` permet de sélectionner les options du débruiteur pour un autre test. À partir de 1 024 jetons image et texte cumulés, le mode par blocs garde le checkpoint ouvert et transfère sur le GPU les poids de l’étape en cours. Il libère ces poids après l’exécution de l’étape et réutilise les graphes compilés pour les pas suivants.

Sur Colab, les sondes ont été compilées avec Bazel 9.3.0 et ZML `0bb81d1957fb6438faddd1751566ad73eed97914`. Les trois checkpoints ComfyUI ont été contrôlés avant le test. Sur la NVIDIA L4, les sorties de l’encodeur et du décodeur VAE avaient pour propriétaire `cuda:0` et les RMSE face à PyTorch étaient respectivement de `6,59e-6` et `2,65e-6`.

Le débruiteur a préparé ses 1 933 tenseurs, dont 839 matrices FP8 à échelle scalaire. Les essais sur le graphe complet ont donné les résultats suivants avec le profil CUDA Q/K et les mêmes entrées.

| Allocation et compilation | Étape atteinte |
| --- | --- |
| VMM par défaut | Budget physique PJRT de 19,83 Gio atteint pendant le chargement |
| Allocations CUDA directes | VRAM épuisée pendant le chargement |
| BFC | Tous les poids chargés. Tampon temporaire de 75,68 Gio demandé à l’exécution |
| BFC avec fusion GEMM Triton | Tous les poids chargés. Tampon temporaire de 75,44 Gio demandé à l’exécution |

L’option `--cuda-low-memory` compile les 60 blocs de débruitage dans des graphes distincts sous Linux. Les poids résidents sont chargés une fois et les états intermédiaires restent sur CUDA. L’allocateur BFC agrandit sa réserve à la demande jusqu’à 98 % du budget mémoire CUDA pour alimenter les graphes. Les tenseurs FP8 à échelles et les calculs FP32 sont conservés. Les mesures de placement indiquent `allocator=bfc` et `graph=blockwise` quand ZML confirme ces réglages. Le rejeu du deuxième pas utilise le même format de tenseurs que le profil sur graphe complet.

Le rejeu par blocs de l’image 64 × 48 a exécuté les deux pas et le VAE final sur la L4. Les deux mises à jour d’Euler ont donné un écart nul avec le rejeu hôte. Les mêmes poids et les mêmes latents sérialisés ont servi pour les mesures suivantes contre PyTorch CUDA.

| Profil par blocs | RMSE vitesse 1 à entrée commune | RMSE vitesse 2 à entrée commune | RMSE latent final | RMSE RGB |
| --- | ---: | ---: | ---: | ---: |
| Standard | 0,15927 | 0,08711 | 0,09326 | 0,12031 |
| CUDA Q/K | 0,15928 | 0,08863 | 0,09325 | 0,12281 |
| Standard avec Triton GEMM désactivé | 0,002294 | 0,000224 | 0,003712 | 0,007233 |
| Standard avec Triton GEMM et cuBLASLt désactivés | 0,002042 | 0,000175 | 0,003481 | 0,007107 |
| CUDA Q/K avec Triton GEMM et cuBLASLt désactivés | 0,002309 | 0,000179 | 0,005774 | 0,010341 |
| Attention ordonnée avec Triton GEMM et cuBLASLt désactivés | 0,002625 | 0,000105 | 0,005031 | 0,009503 |
| Scores centrés avec Triton GEMM et cuBLASLt désactivés | 0,002652 | 0,000184 | 0,007016 | 0,013233 |
| Premier bloc corrigé avec Triton GEMM et cuBLASLt désactivés | 0,000742 | 0,000399 | 0,001455 | 0,002640 |

Le meilleur résultat utilise `XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_enable_cublaslt=false"` et `--cuda-low-memory`. Sous les réglages par défaut de XLA, la RMSE relative du flux image est de `1,90e-4` avant le premier bloc et de `1,84e-2` à sa sortie. Les quatre rotations RoPE mesurées sur les mêmes Q et K ont donné un écart nul.

Avec le profil standard et cuBLAS, PyTorch a reproduit un écart final de `3,494e-3` lorsqu’il a pris le latent ZML comme entrée du deuxième pas. La deuxième vitesse ZML comparée à PyTorch sur ce latent a donné une RMSE de `2,29e-4`. Pour le premier pas, le remplacement des activations d’entrée du premier bloc par celles de PyTorch a ramené la RMSE de la vitesse de `2,042e-3` à `1,889e-3`. La substitution à la sortie du bloc l’a ramenée à `5,273e-5`. Ces rejeux situent l’écart résiduel dans le premier bloc.

Le premier bloc du profil standard à faible mémoire applique désormais le facteur d’échelle aux scores d’attention après le produit Q × Kᵀ, comme l’attention sous-quadratique de ComfyUI. Sur les Q, K et V capturés dans PyTorch CUDA au premier pas, un rejeu du calcul d’attention en FP32 a mesuré une RMSE de `1,181e-3` avec le facteur appliqué à Q et de `2,724e-6` avec le facteur appliqué aux scores.

La projection de modulation de l’image et les projections Q, K et V de l’attention utilisent l’ordre d’accumulation relevé sur la L4. La contraction est découpée en deux partitions, chacune répartie sur quatre accumulateurs qui traitent des groupes de huit coefficients. Les projections Q, K et V suivent ce chemin lorsque chaque flux a au plus 32 jetons. La normalisation de couche suit l’ordre des réductions CUDA et la modulation arrondit la multiplication-addition une seule fois.

Ce profil par blocs a été recompilé dans la session Colab et comparé à PyTorch `2.10.0+cu128` sur la L4, avec le checkpoint FP8 ComfyUI et les mêmes entrées 64 × 48. Le latent du premier pas a donné une RMSE de `3,191e-4`. La seconde vitesse, calculée sur le latent de ZML, a donné `4,038e-3`. Les deux mises à jour d’Euler ont reproduit bit pour bit le calcul hôte à vitesses identiques. Le seuil de `1e-3` est atteint pour les deux vitesses à entrée commune. La trajectoire indépendante reste au-dessus du seuil pour la seconde vitesse, le latent final et le RGB.

### Validation avec l’arithmétique CUDA de référence

Le mode `--cuda-reference-math` reproduit l’arithmétique FP32 de l’encodage du temps et du texte et du premier bloc. Les projections utilisent le budget cuBLASLt qui applique le biais avant la dernière partition. La modulation arrondit le produit en FP32 avant d’ajouter le décalage, conformément aux tenseurs observés dans PyTorch CUDA. ZML exécute les opérations du module sur son flux CUDA et conserve les activations intermédiaires sur le GPU.

La sonde Zig compilée sur Colab a exécuté les deux pas avec le module CUDA compilé par `nvcc 13.0.88`. Le rejeu utilise les mêmes entrées 64 × 48 et les mêmes poids ComfyUI que la comparaison précédente. Les sorties du débruiteur et du VAE ont été vérifiées sur `cuda:0` avec PJRT `cuda`.

| Mesure face à PyTorch CUDA | RMSE |
| --- | ---: |
| Première vitesse | 0,00004650 |
| Latent après le premier pas | 0,00002000 |
| Deuxième vitesse avec le même latent ZML dans les deux moteurs | 0,00001032 |
| Deuxième vitesse sur les trajectoires indépendantes | 0,00059836 |
| Latent final | 0,00021046 |
| RGB final normalisé | 0,00046976 |

Chaque mesure est inférieure à `0,001`. Pour le RGB de ce profil, le script applique le seuil RMSE et affiche aussi l’écart absolu maximal, mesuré à `0,00873` sur les valeurs normalisées dans `[-1, 1]`.

### Validation 512 × 512 sur Colab L4

La sonde a exécuté les deux pas d’Euler avec les poids FP8 ComfyUI, 1 024 jetons pour l’image, 1 024 pour la référence et 80 pour le conditionnement texte. Le débruiteur a chargé les poids bloc par bloc. Le débruiteur et le VAE ZML ont produit leurs sorties sur `cuda:0` avec PJRT `cuda`. PyTorch CUDA a reçu les mêmes entrées et utilisé la même précision FP32 pour les calculs de référence.

| Mesure face à PyTorch CUDA | RMSE |
| --- | ---: |
| Première vitesse | 0,00000186 |
| Latent après le premier pas | 0,00000080 |
| Deuxième vitesse avec le latent PyTorch partagé | 0,00000177 |
| Deuxième vitesse avec le latent ZML partagé | 0,00000171 |
| Deuxième vitesse sur les trajectoires indépendantes | 0,00003379 |
| Latent final | 0,00001197 |
| RGB final normalisé | 0,00001185 |

Les deux mises à jour d’Euler de ZML correspondent bit pour bit à leur rejeu hôte avec les mêmes vitesses. Le RGB final présente un écart absolu maximal de `0,001196` sur les valeurs normalisées dans `[-1, 1]`. Toutes les RMSE sont inférieures au seuil de `0,001`.

### Validation 1024 × 1024 sur Colab L4

Reprendre la commande Linux avec `--width 1024 --height 1024 --vision-size 224`. Le débruiteur reçoit 4 096 jetons pour l’image, 4 096 pour la référence et 80 pour le texte. Les 60 blocs chargent leurs poids au moment de l’exécution et calculent l’attention par groupes de 1 024 requêtes. Les deux moteurs calculent en FP32 avec les mêmes poids ComfyUI. Le débruiteur et le VAE ZML ont produit leurs sorties sur `cuda:0` avec PJRT `cuda`.

| Mesure face à PyTorch CUDA | RMSE |
| --- | ---: |
| Première vitesse | 0,00000211 |
| Latent après le premier pas | 0,00000091 |
| Deuxième vitesse avec le latent PyTorch partagé | 0,00000131 |
| Deuxième vitesse avec le latent ZML partagé | 0,00000122 |
| Deuxième vitesse sur les trajectoires indépendantes | 0,00001790 |
| Latent final | 0,00000642 |
| RGB final normalisé | 0,00000841 |

Les deux mises à jour d’Euler correspondent bit pour bit à leur rejeu hôte. Toutes les mesures du tableau proviennent d’une exécution complète de `validate_linux.py` sur une nouvelle session Colab L4 avec PyTorch `2.10.0+cu128` et passent le seuil de `0,001` RMSE. Le RGB final présente un écart absolu maximal de `0,000697` sur les valeurs normalisées dans `[-1, 1]`. Un rejeu séparé du deuxième pas depuis le latent ZML sauvegardé a donné une RMSE de `0,00000112`. Les journaux sont dans `validation-output`.
