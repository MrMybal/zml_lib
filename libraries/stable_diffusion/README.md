# Stable Diffusion 1.5

Le portage Zig/ZML génère une image à partir d’un prompt texte. Il réutilise l’encodeur CLIP-L et le décodeur VAE de `zml_lib`. Le U-Net reprend les 686 tenseurs CompVis, dont les blocs résiduels et les attentions conditionnées par le texte. Le décodeur charge 140 tenseurs, dont la convolution `post_quant_conv`. L’ordonnanceur DDIM déterministe utilise la grille d’entraînement de SD 1.5. Le lanceur accepte un prompt négatif et une échelle de guidage CFG. Le U-Net reste chargé pour tous les pas.

Les profils texte-vers-image et Inpainting utilisent des poids et des activations FP32. Ils produisent un PNG RGB, de 64 à 512 pixels par dimension, par multiples de 64. Le portage de SDXL reste à faire.

Le convertisseur lit un SafeTensors combiné local au format CompVis, vérifie les clés et dimensions, puis écrit trois fichiers de poids et le manifeste `sd15.json`. Utiliser `v1-5-pruned-emaonly.safetensors` de la révision `f03de32`, SHA-256 `6ce0161689b3853acaa03779ec93eafe75a02f4ced659bee03f50797806fa2fa`. Le dossier du tokenizer CLIP de la même révision contient `vocab.json`, `merges.txt` et sa configuration. Les poids sont soumis à CreativeML Open RAIL-M et restent dans la bibliothèque locale. [Checkpoint et licence](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/f03de32)

## Validation Windows x64

Configurer les DLL du plugin CUDA et du compilateur IREE. Le build utilise l’épingle Windows de [zml.lock.json](../../zml.lock.json). Le Python du lanceur utilise NumPy, Pillow et Transformers. La conversion et le test de référence utilisent aussi PyTorch, SafeTensors et Diffusers. Le test de référence utilise le U-Net, le VAE et DDIM de Diffusers, ainsi que CLIP de Transformers, directement à partir du checkpoint d’origine. [Configuration CUDA](../botsort/README.md#validation-cuda-sous-windows-x64)

```powershell
$env:ZML_TEMP_ROOT = 'D:\AI\zml_lib_Temp'
$py = 'D:\AI\ComfyUI\.venv\Scripts\python.exe'
$flags = @('--@zml//platforms:cuda=true', '--@zml//platforms:rocm=false', '--@zml//platforms:vulkan=false')
$build = @('--target', '//libraries/stable_diffusion:sd15_probe', '--target', '//libraries/clip:clip_text_probe')
$checkpoint = 'D:\models\v1-5-pruned-emaonly.safetensors'
$clipTokenizer = 'D:\models\sd15-tokenizer'
& $py -B .\scripts\build.py @build -- --repo_contents_cache= --jobs=16 --host_compilation_mode=fastbuild @flags `
  --then $py -B .\libraries\stable_diffusion\validate.py --bin-dir '{bin}' --checkpoint $checkpoint `
  --clip-tokenizer $clipTokenizer --expected-device 'RTX 3090' --seed 42 `
  --prompt 'A red lighthouse by the sea at dawn' --negative-prompt 'blur'
```

Le test compare les deux conditionnements CLIP, une prédiction du U-Net, le latent après quatre pas DDIM, le VAE à latent identique et le RGB8 de bout en bout. Avec PyTorch 2.10, Diffusers 0.35.2 et Transformers 4.57.3, les écarts absolus maximaux en 128 × 128 étaient de `4,39e-5` pour CLIP, `7,87e-6` pour le U-Net, `2,67e-4` pour le latent final et `6,14e-6` pour le VAE. Le RGB8 diffère d’au plus 1 niveau sur 255 par canal. Les sorties étaient sur la RTX 3090 via `iree_cuda`.

Pour comparer le PNG du lanceur au PNG exporté par la CLI ImageStudio en 512 × 512, utiliser une copie locale d’ImageStudio qui expose l’opération `generate_sd15_zml`.

```powershell
$studioApp = 'D:\AI\CyTools\CyTools_ImageStudio\app'
$studioPython = 'D:\AI\CyTools\CyTools_ImageStudio\runtime\python\Scripts\python.exe'
& $py -B .\scripts\build.py @build -- --repo_contents_cache= --jobs=16 --host_compilation_mode=fastbuild @flags `
  --then $py -B .\libraries\stable_diffusion\validate_cli.py --bin-dir '{bin}' --checkpoint $checkpoint `
  --clip-tokenizer $clipTokenizer --studio-app $studioApp --studio-python $studioPython
```

Chaque commande crée son dossier de test sous `ZML_TEMP_ROOT` et le supprime à la fin, avec les poids convertis, les sorties et les caches du test. Les scripts acceptent aussi `--weights` pour réutiliser un bundle préparé.

En 512 × 512, avec la graine 42, quatre pas DDIM et un CFG de 7,5, le lanceur et la CLI ImageStudio ont produit le même PNG à l’octet près. Le SHA-256 est `cefa7634d08b5773f73c825088ec504c8086b00e6e30a22f30e7d768e3ed3e05`. Les onze contrôles de placement par parcours ont confirmé `iree_cuda` sur la RTX 3090. Le lanceur a pris 61,38 s et la CLI 61,09 s sur ce poste.

## Appel direct

Avec les binaires déjà disponibles, préparer le checkpoint et appeler le lanceur. `$weights` est un nouveau dossier pour les poids convertis et `$png` désigne le fichier de sortie demandé.

```powershell
& $py -B .\libraries\stable_diffusion\prepare.py $checkpoint --bin-dir $bin --output $weights
& $py -B .\libraries\stable_diffusion\generate.py --bin-dir $bin --weights $weights `
  --clip-tokenizer $clipTokenizer --prompt 'A red lighthouse by the sea at dawn' --negative-prompt 'blur' `
  --output $png --backend cuda --expected-device 'RTX 3090' --width 512 --height 512 --steps 4 --seed 42
```

Le lanceur retourne un objet JSON sur stdout et écrit ses diagnostics sur stderr. Il vérifie le périphérique propriétaire des sorties avant leur copie vers l’hôte. Les fichiers d’entrée texte UTF-8 peuvent être passés via `--prompt-file` et `--negative-prompt-file`.

## SD 1.5 Inpainting

Le profil `sd15-inpaint` utilise un U-Net à neuf canaux. À chaque pas, ZML concatène le latent à débruiter, un canal du masque à la résolution latente et les quatre canaux du latent de l’image masquée. L’encodeur VAE ZML charge 108 tenseurs et produit les moments du posterior. À force partielle, il encode aussi l’image source et DDIM ajoute le bruit au premier pas retenu. Le nombre de pas appliqués est `int(steps × strength)`. À force 1, le débruitage commence au premier pas, à partir du bruit seul.

L’image RGB et le masque en niveaux de gris doivent avoir les mêmes dimensions. Les valeurs du masque de 128 à 255 désignent la zone à retoucher. Le PNG final recopie exactement les pixels RGB de la source en dehors de cette zone. La graine détermine le bruit de diffusion et celui du posterior du VAE.

Utiliser le checkpoint local `sd-v1-5-inpainting.ckpt` de la révision `8a4288a76071f7280aedbdb3253bdb9e9d5d84bb`, SHA-256 `c6bbc15e3224e6973459ba78de4998b80b50112b0ae5b5c67113d56b4e366b19`. Les poids sont sous licence CreativeML Open RAIL-M. Le convertisseur lit le `.ckpt` avec `torch.load(weights_only=True)` et accepte aussi le même checkpoint au format SafeTensors. Le fichier `encoder.safetensors` s’ajoute aux trois fichiers du bundle texte-vers-image. [Checkpoint et licence](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting/tree/8a4288a76071f7280aedbdb3253bdb9e9d5d84bb)

```powershell
$inpaintCheckpoint = 'D:\models\sd-v1-5-inpainting.ckpt'
$inpaintTokenizer = 'D:\models\sd15-inpaint-tokenizer'
& $py -B .\libraries\stable_diffusion\prepare.py $inpaintCheckpoint --profile sd15-inpaint --bin-dir $bin --output $inpaintWeights
& $py -B .\libraries\stable_diffusion\inpaint.py --bin-dir $bin --weights $inpaintWeights --clip-tokenizer $inpaintTokenizer `
  --image $image --mask $mask --prompt 'A red lighthouse by the sea at dawn' --negative-prompt 'blur' `
  --output $png --backend cuda --expected-device 'RTX 3090' --steps 4 --strength 0.75 --seed 42
```

### Comparaison PyTorch et CLI sous Windows

Le premier test compare les sorties de CLIP, du VAE (encodage et décodage) et du U-Net à celles de Transformers et Diffusers FP32, avec les mêmes tirages de bruit. Il compare aussi le latent après DDIM et l’image brute puis composée avec la source. Le second test compare le résultat à l’octet près avec la CLI ImageStudio et vérifie la retouche dans le masque ainsi que la conservation des pixels hors masque.

```powershell
& $py -B .\scripts\build.py @build -- --repo_contents_cache= --jobs=16 --host_compilation_mode=fastbuild @flags `
  --then $py -B .\libraries\stable_diffusion\validate_inpaint.py --bin-dir '{bin}' --checkpoint $inpaintCheckpoint `
  --clip-tokenizer $inpaintTokenizer --expected-device 'RTX 3090' --seed 42 `
  --prompt 'A red lighthouse by the sea at dawn' --negative-prompt 'blur'
& $py -B .\scripts\build.py @build -- --repo_contents_cache= --jobs=16 --host_compilation_mode=fastbuild @flags `
  --then $py -B .\libraries\stable_diffusion\validate_cli.py --inpaint --bin-dir '{bin}' --checkpoint $inpaintCheckpoint `
  --clip-tokenizer $inpaintTokenizer --studio-app $studioApp --studio-python $studioPython
```

Les scripts génèrent leur image et leur masque sous `ZML_TEMP_ROOT` et les suppriment en fin de test. Pour comparer une autre paire d’images à PyTorch, passer `--image` et `--mask` à `validate_inpaint.py`. Ce script accepte aussi `--strength 1` pour vérifier le départ à partir du bruit seul.

En 192 × 128, la RTX 3090 via `iree_cuda` retrouve exactement le masque latent de PyTorch aux forces 0,75 et 1. L’image composée diffère d’au plus 1 niveau sur 255 par canal pour les deux forces. Chaque sortie modifie 2 424 pixels dans le masque et zéro hors masque.

| Écart absolu maximal face à PyTorch | Force 0,75 · 3 pas DDIM | Force 1 · 4 pas DDIM |
| --- | ---: | ---: |
| Latent de l’image masquée | `1,43e-4` | `1,43e-4` |
| Latent de l’image source | `1,33e-5` | — |
| Première prédiction du U-Net | `5,36e-6` | `7,87e-6` |
| Latent final | `1,89e-4` | `1,10e-4` |
| Décodeur VAE à latent identique | `2,62e-5` | `2,32e-5` |

En 512 × 512 à force 0,75, le lanceur ZML et la CLI ImageStudio produisent le même PNG octet pour octet. Les onze contrôles de placement de chaque parcours confirment la RTX 3090 via `iree_cuda`. Le test compte 20 729 pixels modifiés dans le masque et zéro hors masque. Le SHA-256 du PNG est `8588ef4401c511897752f8b81c38cf57449384331421fb1dd250ec84fd2fdb36`.
