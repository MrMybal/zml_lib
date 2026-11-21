# SAM 2.0 Hiera base+

Le port Zig utilise le checkpoint ComfyUI `sam2_hiera_base_plus.safetensors`. L’encodeur Hiera à 24 blocs applique les changements d’échelle des requêtes et la fusion FPN. Le décodeur partage le code d’encodage des prompts et d’attention à deux voies avec la bibliothèque SAM, avec ses propres poids. Il ajoute les deux branches spatiales haute résolution et un score de présence d’objet. Le profil vidéo utilise l’encodeur de mémoire et l’attention rotative pour suivre un objet dans l’ordre des images. [Référence SAM 2](https://github.com/facebookresearch/sam2)

L’encodeur reçoit une image RGB redimensionnée à 1 024 × 1 024 et normalisée avec les paramètres ImageNet. Il fournit, dans l’ordre, les tenseurs NCHW `[1,256,64,64]`, `[1,32,256,256]` et `[1,64,128,128]`. Le décodeur réutilise ces tenseurs. Après les embeddings, le paquet contient un indicateur de masque, un masque `256 × 256` ou des zéros, les triplets `(x, y, label)` puis l’indicateur de mémoire. La valeur `1` sélectionne l’embedding initial, et `0` indique que la première carte a été conditionnée par la mémoire temporelle. Le décodeur renvoie quatre masques `256 × 256`, quatre scores, quatre tokens de 256 valeurs et le logit de présence.

`memory.Encoder` reçoit la carte primaire non conditionnée, le masque à 1 024 × 1 024, le token sélectionné, le logit de présence et l’indicateur de masque issu d’un prompt. Il renvoie une carte mémoire `[1,64,64,64]` suivie du pointeur de 256 valeurs. `memory.Attention` reçoit la carte primaire de l’image courante, jusqu’à sept blocs contenant l’indice temporel puis la carte mémoire, et jusqu’à seize pointeurs. Le test vidéo montre la mise en file de l’image de conditionnement, les six dernières mémoires propagées et l’ordre des pointeurs. [Préparation des entrées vidéo](video_validation.py)

Le convertisseur prépare quatre fichiers SafeTensors. Il interpole une fois le fond positionnel appris pour le profil fixe de 1 024 × 1 024 et matérialise les phases spatiales et rotatives constantes. Le schéma et les poids d’origine sont vérifiés par le test PyTorch.

## Rejeu Windows x64

    $py = (Join-Path $env:COMFYUI_ROOT '.venv\Scripts\python.exe')
    $original = (Join-Path $env:COMFYUI_ROOT 'models\sam2\sam2_hiera_base_plus.safetensors')
    $reference = (Join-Path $env:COMFYUI_ROOT 'custom_nodes\comfyui-segment-anything-2')
    $image = (Join-Path $env:COMFYUI_ROOT '.venv\Lib\site-packages\matplotlib\mpl-data\sample_data\grace_hopper.jpg')
    $work = 'validation-output\sam2-replay'
    $env:ZML_CPU_PJRT_DLL = (Resolve-Path '..\zml_win\bazel-win-tools\pjrt-cpu\pjrt_cpu.dll').Path
    & $py .\libraries\sam2\validate.py convert --checkpoint $original --output "$work\weights"
    if ($LASTEXITCODE) { throw 'Conversion SAM 2 en échec' }
    python .\scripts\build.py -- --jobs=8 //libraries/sam2:sam2_probe --then $py .\libraries\sam2\validate.py compare --video --checkpoint $original --weights "$work\weights" --reference $reference --image $image --probe '{bin}\libraries\sam2\sam2_probe.exe' --backend cpu --output "$work\cpu"
    if ($LASTEXITCODE) { throw 'Comparaison SAM 2 en échec' }

Le test image compare les trois cartes Hiera/FPN, les logits et les scores pour deux jeux de prompts sur le même embedding. Avec `--video`, il suit ensuite un objet sur trois images, avec un seul point initial, et compare les cartes mémoires, les pointeurs et les masques propagés à la référence FP32.

Le rejeu sur `cpu:0` sous Windows x64 a donné une IoU de `1,0` pour les masques des deux jeux de prompts et pour les trois images de la séquence. L’écart absolu maximal des logits du masque vidéo est de `2,28e-4`, et celui de la carte conditionnée par la mémoire de `1,02e-4`. Le suivi testé propage un objet vers l’avant.
