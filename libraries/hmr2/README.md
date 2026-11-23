# HMR2 pour GVHMR

Dans GVHMR, `--crop-probe` calcule la mise à demi-résolution, le recadrage et la normalisation ImageNet dans le même graphe ZML et partage le résultat avec ViTPose-H. [Prétraitement commun et commande PowerShell](../gvhmr/README.md#prétraitement-vidéo-par-zml)

Le port calcule le jeton d’apparence `[N,1024]` que GVHMR reçoit de HMR2. Il partage l’encodeur ViT-H/16 de ViTPose et exécute les six blocs du décodeur Transformer avec les poids propres à HMR2. Ce profil couvre `feat_mode=True`, utilisé pour extraire les caractéristiques vidéo. [4DHumans](https://shubham-goel.github.io/4dhumans/) · [Profil HMR2 de GVHMR](https://github.com/zju3dv/GVHMR/blob/6ec3ca39336c50492c0fae65fba2fb831fc7d866/hmr4d/network/hmr2/hmr2.py)

`hmr2_probe.exe schema` affiche les 494 tenseurs FP32 attendus. Le convertisseur sélectionne le backbone et `smpl_head.transformer` dans le `state_dict` local `epoch=10-step=25000.ckpt`. [Convertisseur](../../scripts/pose_checkpoint.py)

```powershell
& $py -B .\scripts\pose_checkpoint.py --model hmr2 --probe $probe --checkpoint $checkpoint --weights $safetensors
& $probe cuda $safetensors $frames $batch $inputRaw $outputRaw 2
```

Le lanceur reçoit les mêmes recadrages RGB normalisés, dimensions `[N,3,256,192]`, que [ViTPose-H](../vitpose/README.md). L’entrée et la sortie sont des buffers FP32 little-endian contigus, avec un jeton par image. [Validation dans GVHMR](../gvhmr/README.md#prétraitement-vidéo-par-zml)
