# Stable Diffusion 1.5

Le portage Zig/ZML génère une image à partir d’un prompt texte. Il réutilise l’encodeur CLIP-L et le décodeur VAE de `zml_lib`. Le U-Net reprend les 686 tenseurs CompVis, dont les blocs résiduels et les attentions conditionnées par le texte. Le décodeur charge 140 tenseurs, dont la convolution `post_quant_conv`. L’ordonnanceur DDIM déterministe utilise la grille d’entraînement de SD 1.5. Le lanceur accepte un prompt négatif et une échelle de guidage CFG. Le U-Net reste chargé pour tous les pas.

Le profil utilise des poids et des activations FP32. Il produit un PNG RGB, de 64 à 512 pixels par dimension, par multiples de 64. Il couvre le checkpoint texte-vers-image à quatre canaux. Le checkpoint SD 1.5 Inpainting à neuf canaux et SDXL restent à porter.

Le convertisseur lit un SafeTensors combiné local au format CompVis, vérifie les clés et dimensions, puis écrit trois fichiers de poids et le manifeste `sd15.json`. Utiliser `v1-5-pruned-emaonly.safetensors` de la révision `f03de32`, SHA-256 `6ce0161689b3853acaa03779ec93eafe75a02f4ced659bee03f50797806fa2fa`. Le dossier du tokenizer CLIP de la même révision contient `vocab.json`, `merges.txt` et sa configuration. Les poids sont soumis à CreativeML Open RAIL-M et restent dans la bibliothèque locale. [Checkpoint et licence](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/f03de32)

## Validation Windows x64

Configurer les DLL du plugin CUDA et du compilateur IREE. Le build utilise l’épingle Windows de [zml.lock.json](../../zml.lock.json). Le Python du lanceur utilise NumPy, Pillow et Transformers. La conversion et le test de référence utilisent aussi PyTorch, SafeTensors et Diffusers. Le test de référence utilise le U-Net, le VAE et DDIM de Diffusers, ainsi que CLIP de Transformers, directement à partir du checkpoint d’origine. [Configuration CUDA](../botsort/README.md#validation-cuda-sous-windows-x64)

```powershell
$env:ZML_TEMP_ROOT = 'D:\_Project\zml_lib_Temp'
$py = 'D:\ComfyUI\.venv\Scripts\python.exe'
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
$studioApp = 'D:\_Project\_CyTools\CyTools_ImageStudio\app'
$studioPython = 'D:\_Project\_CyTools\CyTools_ImageStudio\runtime\python\Scripts\python.exe'
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
