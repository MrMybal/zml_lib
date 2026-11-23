# FLUX.1 Schnell

Le débruiteur Zig/ZML reprend les 19 blocs à deux flux et les 38 blocs à flux unique de Schnell. Le graphe combine le conditionnement CLIP-L, les états cachés de T5 XXL, l’embedding temporel et les positions rotatives de la grille. Le lanceur conserve le modèle chargé pendant les pas d’Euler. [Implémentation de Black Forest Labs](https://github.com/black-forest-labs/flux/tree/802fb4713906133fcbd0d8dc5351620ca4773036)

Le profil `e4m3` lit les octets SafeTensors E4M3FN et les conserve sur le périphérique. Le graphe décode leurs valeurs en FP32 pour les calculs. `f32` et `f16` sont aussi disponibles. La conversion extrait du checkpoint combiné ComfyUI Schnell le débruiteur, les encodeurs texte et le décodeur du VAE officiel. Elle écrit les tenseurs un à un. Le lanceur peut aussi utiliser TAEF1 pour les prévisualisations. [Checkpoint ComfyUI, Apache 2.0](https://huggingface.co/Comfy-Org/Flux-1-Schnell/blob/0cb207e/flux1-schnell-fp8.safetensors) · [TAEF1](https://github.com/madebyollin/taesd)

## Lanceur Windows x64

Configurer les DLL du backend choisi. Pour CUDA, utiliser les variables du plugin IREE et du compilateur. [Configuration CUDA](../botsort/README.md#validation-cuda-sous-windows-x64)

```powershell
$env:ZML_TEMP_ROOT = 'D:\_Project\zml_lib_Temp'
$py = 'D:\ComfyUI\.venv\Scripts\python.exe'
$flags = @('--@zml//platforms:cuda=true', '--@zml//platforms:rocm=false', '--@zml//platforms:vulkan=false')
& $py -B .\scripts\build.py --target //libraries/flux1:flux1_probe --target //libraries/clip:clip_text_probe `
  --target //libraries/t5:t5_encoder --target //libraries/taesd:taef1_probe `
  -- --repo_contents_cache= --jobs=16 --host_compilation_mode=fastbuild @flags
$bin = (Resolve-Path .\validation-output\bin).Path
& $py -B .\libraries\flux1\prepare.py --bin-dir $bin --source $checkpoint --taef1 $taef1 --output $weights
& $py -B .\libraries\flux1\generate.py --bin-dir $bin --weights $weights `
  --clip-tokenizer $clipTokenizer --t5-tokenizer $t5Tokenizer --prompt 'A red lighthouse by the sea at dawn.' `
  --output $png --backend cuda --expected-device 'RTX 3090' --steps 4 --decoder vae
```

`$checkpoint` désigne le SafeTensors combiné et `$weights` un nouveau dossier. L’option de conversion `--taef1 $taef1` ajoute le fichier local `taef1_decoder.pth`, que le lanceur utilise avec `--decoder taef1`. Les tokenizers CLIP-L et T5 v1.1 sont des dossiers locaux. `$png` est un nouveau fichier dans un dossier existant. Le lanceur écrit ses diagnostics sur stderr et retourne le chemin, les dimensions, la graine, le nombre de pas et le placement en JSON sur stdout.

Un appelant peut fournir un fichier texte UTF-8 via `--prompt-file` ou passer le texte via `--prompt`. `--workspace` place les fichiers temporaires dans un sous-dossier unique du job, supprimé à la sortie normale du lanceur. ImageStudio utilise ces deux options pour traiter les requêtes `generate_flux_zml` de son interface, de sa CLI et de MCP. La procédure de configuration figure dans `app/README.md` du dépôt CyTools_ImageStudio.

### FP8 à échelles séparées

`prepare.py` détecte le profil `e4m3_scaled` dans un checkpoint combiné ou dans un débruiteur BFL/ComfyUI fourni avec `--denoiser`. Dans le second cas, `--source` fournit les encodeurs et le VAE du checkpoint Schnell combiné. Le convertisseur accepte une échelle de poids scalaire positive par matrice, enregistrée sous `.weight_scale` ou `.scale_weight`. Les marqueurs `scaled_fp8`, `.comfy_quant` et les métadonnées `_quantization_metadata` permettent de reconnaître les variantes ComfyUI. [Format de quantification ComfyUI](https://github.com/Comfy-Org/ComfyUI/blob/master/QUANTIZATION.md)

ZML garde les codes E4M3FN sur le périphérique et applique leurs échelles dans la table de décodage FP32. Les tenseurs ordinaires du débruiteur sont conservés en FP32 dans le bundle. Le calcul utilise des activations FP32, pour lesquelles `input_scale` reste une donnée de calibration. Le résultat JSON indique le profil `storage`. Pour ImageStudio, affecter le dossier préparé à `weights` dans la configuration FLUX.

Pour tester la déquantification contre PyTorch, ajouter `--storage e4m3_scaled` à `validate.py` avec le profil `fixture`. Dans la commande de validation des résolutions ci-dessous, ajouter `--repack-scaled --size 1024x1024` pour reproduire l’essai depuis le checkpoint Schnell FP8 ordinaire. Ce reconditionnement de test préserve exactement la valeur déquantifiée de chaque poids et ajoute des échelles différentes de `1`.

Sur la RTX 3090, le modèle réduit diffère de PyTorch d’au plus `9,54e-7` avec FP8 à échelles et de `1,08e-6` avec un LoRA ajouté. En 1024 × 1024, la CLI ImageStudio a utilisé le profil `e4m3_scaled` avec 312 matrices à échelle non unitaire. Les huit contrôles de placement ont confirmé `iree_cuda`. Avec la graine 42, 256 tokens T5, quatre pas d’Euler et le VAE officiel, le PNG a reproduit le SHA-256 du checkpoint FP8 de référence, `58af39e060b14a98d66cfa7778103d147a9a541c4937095a060bf3f0577dfb8c`. La génération a pris 321,95 s et la VRAM totale utilisée sur la carte a culminé à 21 316 Mio.

### LoRA du débruiteur

Ajouter `--lora $lora 0.8` à la commande `generate.py`, où `$lora` désigne un fichier `.safetensors` local. Répéter l’option pour empiler plusieurs LoRA. Une intensité de `0` désactive l’adaptateur. Le lanceur accepte les paires linéaires `lora_down` / `lora_up` et `lora_A` / `lora_B` sur les modules natifs BFL ou leurs équivalents Diffusers. Les projections Q, K, V et MLP séparées sont appliquées aux tranches correspondantes des matrices fusionnées. [Mappage FLUX de ComfyUI](https://github.com/Comfy-Org/ComfyUI/blob/88ab4a06566454ad89db8f0bedb970d6c08cd1b7/comfy/utils.py#L677)

ZML calcule l’ajout `intensité × (alpha / rang) × B(A(x))` en FP32 à chaque couche ciblée. En l’absence d’alpha, le rapport `alpha / rang` vaut `1`. Les facteurs sont chargés une fois pour tous les pas d’Euler. Les poids FP8 de base conservent leurs valeurs. Le résultat JSON indique l’intensité et le nombre de cibles par adaptateur. Les entrées visées sont les couches linéaires du débruiteur Schnell.

Le test `validate.py --probe $probe --upstream $fluxSrc --backend cuda --expected-device 'RTX 3090' --fixture-lora` compare une prédiction et quatre pas d’Euler à PyTorch FP32 après fusion dense des facteurs. Le jeu de test couvre le placement des projections, l’intensité négative, l’empilement, l’alpha et l’ordre des coefficients de la modulation finale.

Sous Windows x64, l’écart absolu maximal de ce test face à PyTorch est de `5,96e-7` sur `cpu:0` et de `8,94e-7` sur la RTX 3090. Avec le checkpoint Schnell E4M3FN et le LoRA Octree à `0,8`, le lanceur direct et la CLI ImageStudio ont appliqué les 494 cibles et produit le même PNG RGB de 96 × 64. Les huit contrôles de placement de chaque parcours ont confirmé `iree_cuda`. La graine était `42`, avec 16 tokens T5, quatre pas d’Euler et le VAE officiel. Le PNG a pour SHA-256 `03b235bfa1164047c4ac37c28a492f404d7ae713df777172b6e87bb8e4dcadc2`. [LoRA Octree sous Apache 2.0, SHA-256 des poids `142d5787f43a510db30513f56fca490f3c6637778a6bc88900a449e0c6f9fb15`](https://huggingface.co/Octree/flux-schnell-lora/blob/c395d0295c36b8e0b363e68285ae903b85e7c704/flux-schnell-lora.safetensors)

À intensité `0`, la CLI a retrouvé le SHA-256 du PNG de référence sans LoRA, `ea5d7757df0f4578b7d32874dca0531de7f5d2bbbf5a7427ddf239857531c423`. À intensité `0,8`, 6 141 des 6 144 pixels ont changé.

En 1024 × 1024 avec 256 tokens T5 et les mêmes réglages, ZML direct et la CLI ImageStudio ont produit le même PNG à l’octet près. Les huit contrôles par parcours ont confirmé la RTX 3090 via `iree_cuda` et chaque parcours a appliqué les 494 cibles du LoRA. Le SHA-256 est `4ff25c07515a45ce2d799b3ec56d9ce2c7e6c53b9575ac174a78c37db798ae98`. Le lanceur direct a pris 312,14 s et la CLI 330,72 s. La VRAM totale utilisée sur la carte a culminé respectivement à 23 316 et 23 016 Mio.

Le rejeu 1024 × 1024 avec le bundle reconditionné sans perte `e4m3_scaled` et Octree à `0,8` a reproduit ce même SHA-256. La comparaison des deux PNG confirme des octets identiques et zéro pixel différent. Chaque parcours a consigné 312 échelles non unitaires, 494 cibles LoRA et huit contrôles de placement sur la RTX 3090 via `iree_cuda`. Le lanceur direct a pris 413,78 s et la CLI 429,03 s. Les pics de VRAM sur la carte étaient de 24 011 et 23 332 Mio respectivement.

## Protocole du débruiteur

`flux1_probe.exe PROFIL STOCKAGE BACKEND POIDS LIGNES COLONNES TOKENS PAS ENTREE SORTIE [MANIFESTE_LORA FACTEURS_LORA]` accepte des fichiers FP32 little-endian. L’entrée contient successivement la grille de latents `[1,LIGNES,COLONNES,64]`, le texte T5 `[1,TOKENS,4096]`, le vecteur CLIP-L `[1,768]` et `PAS+1` valeurs temporelles décroissantes. Le profil de test `fixture` utilise une largeur texte de 32. L’appelant fournit et possède les fichiers. Le lanceur Python convertit les LoRA dans les deux fichiers optionnels du probe. À zéro pas, la sortie contient la vitesse prédite pour l’unique temps. Avec un ou plusieurs pas, la sortie contient le latent après intégration. Le programme vérifie le périphérique de chaque sortie avant sa copie vers l’hôte.

La commande `flux1_probe.exe vae BACKEND POIDS HAUTEUR LARGEUR LATENT RGB` décode un latent planaire `[1,16,H,W]` en couleurs brutes `[1,3,8H,8W]`, en FP32 little-endian. Elle applique l’échelle et le décalage officiels de Schnell.

`validate.py --probe $probe --upstream $fluxSrc --backend cuda --expected-device 'RTX 3090'` compare une prédiction et quatre pas d’Euler au code PyTorch de Black Forest Labs. `$fluxSrc` doit contenir le dossier Python `flux` de la révision indiquée plus haut. Pour le modèle complet, ajouter `--profile schnell --weights $denoiser` avec le fichier `flux.safetensors` du dossier préparé. Ajouter `--vae-weights $vae` pour vérifier le décodeur.

## Validation Windows x64

Le checkpoint ComfyUI utilisé est `0cb207e`, SHA-256 `ead426278b49030e9da5df862994f25ce94ab2ee4df38b556ddddb3db093bf72`. Le débruiteur complet a vérifié ses 776 tenseurs face à Black Forest Labs. Sur la RTX 3090 via `iree_cuda`, l’écart absolu maximal face à PyTorch FP32 est de `6,53e-6` pour une prédiction de vitesse et de `5,48e-6` après quatre pas d’Euler.

Le VAE officiel a vérifié ses 138 tenseurs et restitué le RGB brut attendu. L’écart maximal est de `5,36e-6` sur `cpu:0` et de `4,53e-6` sur la RTX 3090. Les profils réduits du débruiteur passent également sur les deux périphériques.

Le lanceur a produit un PNG RGB de 96 × 64 pixels à partir du prompt de l’exemple, avec 16 tokens T5, la graine 42 et quatre pas. CLIP, T5, le débruiteur et le VAE officiel ont exécuté leurs graphes sur la RTX 3090 via `iree_cuda`. Le fichier produit a pour SHA-256 `ea5d7757df0f4578b7d32874dca0531de7f5d2bbbf5a7427ddf239857531c423`.

Le rejeu avec TAEF1 a aussi produit un PNG RGB de 96 × 64 sur la RTX 3090. Son SHA-256 est `b85b6f3d02a91ba0a1d4c9879e2f886f2303bd91796acff7cf5a59fc27ac2249`.

L’opération ImageStudio a reproduit les deux empreintes avec les mêmes entrées sous Windows x64. Le VAE officiel a été validé depuis la CLI avec export du PNG et TAEF1 depuis MCP stdio avec huit vérifications de placement sur la RTX 3090. Les espaces des jobs ont été supprimés après l’export CLI ou la fin de la session MCP.

### Mesurer les résolutions supérieures

`validate_generation.py` prépare le checkpoint local, valide le PNG et vérifie le placement de chaque étape. Par défaut, il génère en 512 × 512 puis en 1024 × 1024, avec 256 tokens T5, quatre pas d’Euler, le VAE officiel et la graine 42. Les fichiers d’essai sont supprimés après chaque cas. La mémoire rapportée par `nvidia-smi` est celle de la carte entière, y compris les autres applications. L’écart au relevé avant l’inférence est une estimation de la consommation du pipeline.

```powershell
& $py -B .\scripts\build.py --target //libraries/flux1:flux1_probe --target //libraries/clip:clip_text_probe `
  --target //libraries/t5:t5_encoder -- --repo_contents_cache= --jobs=16 --host_compilation_mode=fastbuild @flags `
  --then $py -B .\libraries\flux1\validate_generation.py --bin-dir '{bin}' --checkpoint $checkpoint `
  --clip-tokenizer $clipTokenizer --t5-tokenizer $t5Tokenizer --expected-device 'RTX 3090' --gpu 0
```

Répéter `--size LARGEURxHAUTEUR` pour choisir les résolutions. Chaque ligne JSON indique la durée, le pic VRAM observé, l’empreinte SHA-256 du PNG et le placement CUDA.

Sur la RTX 3090 sous Windows x64, la CLI ImageStudio a généré le PNG 512 × 512 avec ses réglages par défaut et la graine 42 en 167 s. Le lanceur ZML direct a reproduit son SHA-256 `8394dbe5f421ff801dbd8582df227c3f5b95734b6ba7a5bc8a6d86708bf8b119` en 153,36 s. Les pics sur la carte étaient respectivement de 19 161 et 19 109 Mio, soit environ 12 Gio au-dessus des relevés avant génération.

En 1024 × 1024, le lanceur ZML direct a produit un PNG RGB avec le VAE officiel en 471,58 s avec les mêmes réglages. Le pic mesuré sur la carte est de 23 191 Mio, pour un niveau initial de 6 905 Mio. Le SHA-256 du PNG est `58af39e060b14a98d66cfa7778103d147a9a541c4937095a060bf3f0577dfb8c`.

La CLI ImageStudio a exporté le même PNG 1024 × 1024 en 285,14 s. Le journal exporté confirme huit contrôles de placement sur la RTX 3090 via `iree_cuda`. La mémoire totale utilisée sur la carte est passée de 5 980 à 23 301 Mio au pic. L’empreinte SHA-256 est identique à celle du lanceur direct.
