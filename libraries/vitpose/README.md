# ViTPose-H multi-COCO

Dans GVHMR, `--crop-probe` calcule la mise à demi-résolution, le recadrage et la normalisation ImageNet dans le même graphe ZML et partage le résultat avec HMR2. [Prétraitement commun et commande PowerShell](../gvhmr/README.md#prétraitement-vidéo-par-zml)

Le port exécute le ViT-H/16 à 32 blocs, les deux convolutions transposées de la tête COCO et la fusion avec l’image retournée horizontalement. Il reçoit des recadrages RGB normalisés selon ImageNet en FP32 `[N,3,256,192]` et produit les cartes COCO `[N,17,64,48]`. GVHMR peut obtenir directement les coordonnées et scores `[N,17,3]` grâce au décodeur ZML. [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) · [Tête utilisée dans GVHMR](https://github.com/zju3dv/GVHMR/blob/6ec3ca39336c50492c0fae65fba2fb831fc7d866/hmr4d/utils/preproc/vitpose.py)

`vitpose_probe.exe schema` affiche les noms et dimensions des 401 tenseurs FP32. Le convertisseur lit le `state_dict` local `vitpose-h-multi-coco.pth` et ignore les deux compteurs d’entraînement BatchNorm. [Convertisseur](../../scripts/pose_checkpoint.py)

```powershell
& $py -B .\scripts\pose_checkpoint.py --model vitpose --probe $probe --checkpoint $checkpoint --weights $safetensors
& $probe cuda $safetensors $frames $batch $inputRaw $outputRaw 2
```

Le lanceur ouvre une session, charge les poids une fois, traite des lots de taille fixe et complète le dernier lot avec des zéros si nécessaire. Les chemins d’entrée et de sortie désignent des FP32 little-endian contigus, rangés par image. Le fichier de sortie doit être nouveau. Seules les images fournies sont écrites. La session libère les buffers du périphérique avant la sortie. [Validation dans GVHMR](../gvhmr/README.md#prétraitement-vidéo-par-zml)

## Décodage UDP/DARK

Le décodeur retient le premier maximum de chaque carte, applique le filtre gaussien séparable de taille 11, puis affine la position à partir des dérivées de la carte logarithmique. La résolution de la Hessienne (2 × 2) utilise FP64. La projection remet les coordonnées dans la vidéo à partir du centre et de la taille du recadrage. Le score brut est conservé. Pour un score non positif, la position `(-1,-1)` dans la carte est projetée sans raffinement. [UDP](https://openaccess.thecvf.com/content_CVPR_2020/html/Huang_The_Devil_Is_in_the_Details_Delving_Into_Unbiased_Data_CVPR_2020_paper.html) · [DARK](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Distribution-Aware_Coordinate_Representation_for_Human_Pose_Estimation_CVPR_2020_paper.html)

```text
vitpose_probe.exe joints cuda weights.safetensors frames batch input.f32 output.f32 repeats
vitpose_probe.exe verify-joints cuda weights.safetensors frames batch input.f32 output.f32 repeats
vitpose_probe.exe udp cuda frames batch input.f32 output.f32 repeats
```

Pour `joints` et `verify-joints`, chaque ligne d’entrée contient le recadrage NCHW aplati suivi de `[centre_x, centre_y, taille]` en pixels vidéo. Le mode `udp` lit les cartes aplaties suivies des trois mêmes valeurs. Les entrées sont finies et la taille du recadrage est positive. Les sorties contiennent 17 triplets `[x, y, score]` par image. En mode `verify-joints`, les cartes non modifiées suivent ces triplets dans chaque ligne. Le mode `joints` réalise la prédiction et le décodage dans le même graphe et ne transfère que les triplets vers l’hôte.

Avec les variables Windows et CUDA décrites pour GVHMR, le test suivant utilise 7 jeux de cartes synthétiques sans checkpoint. Il vérifie les pics sous-pixel, les bords, les maxima ex æquo et les articulations absentes.

Sous Windows x64 sur RTX 3090 via `iree_cuda`, le test a vérifié 119 articulations, dont deux absentes, avec un écart maximal de `0,000031 px` face à la référence. Sur trois recadrages synthétiques avec le checkpoint CyTools, les modes `joints` et `verify-joints` ont donné les mêmes triplets. [Mesures](port.json)

```powershell
& $py -B .\scripts\build.py --target //libraries/vitpose:vitpose_probe `
  -- --jobs=12 --host_compilation_mode=fastbuild @cuda --then $py -B .\libraries\vitpose\validate_udp.py `
  --probe '{bin}\libraries\vitpose\vitpose_probe.exe' --upstream $upstream --backend cuda --expected-device 'RTX 3090'
```
