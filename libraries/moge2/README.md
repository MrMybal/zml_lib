# MoGe‑2 pour LocalMesh

Le graphe réutilise DINOv2 ViT-L/14 et ses sorties des blocs 5, 11, 17 et 23.
Il calcule le cou partagé et les têtes de points 3D, de masque, de normales
et d’échelle. Le fork utilise la carte de normales pour son relief et la
carte de points et le masque pour son estimation de focale. L’optimisation
de la focale reste celle de LocalMesh. [Code MoGe](https://github.com/microsoft/MoGe)

| Profil | Poids officiels | Révision |
| --- | --- | --- |
| camera | Ruicheng/moge-2-vitl | 39c4d5e957afe587e04eec59dc2bcc3be5ecd968 |
| normal | Ruicheng/moge-2-vitl-normal | cb0e8bbd6b1e243589717c78e750b1ba4c093acf |

Compiler `//libraries/moge2:moge2_probe`. Fournir le `model.pt` local.
La conversion vérifie les 483 ou 521 tenseurs FP32 attendus et la
configuration exacte du modèle.

```bash
python -B libraries/moge2/prepare.py --probe "$probe" --profile normal \
  --checkpoint "$model_pt" --output "$weights"
python -B libraries/moge2/validate.py --probe "$probe" --profile normal \
  --weights "$weights" --original "$model_pt" --upstream "$localmesh" \
  --image "$photo" --scratch "$scratch" --report "$report" \
  --size 384 512 --tokens 3600 --expected-device cuda:0 --expected-pjrt cuda
```

Le probe attend une entrée FP32 little-endian `[1,3,H,W]` en RGB dans [0,1].
La grille ViT reçoit les dimensions calculées par l’appelant à partir de
3 600 jetons et du rapport largeur/hauteur. Le profil normal rend `[H,W,3]`.
Le profil camera range successivement les points `[H,W,3]`, le masque
`[H,W]` et l’échelle scalaire. Le profil normal-full insère la carte des
normales avant le scalaire et utilise le même fichier de poids que normal.

Pour les poids locaux et l’intégration, voir le [suivi du portage](../localmesh_multivue/README.md).
[Poids caméra](https://huggingface.co/Ruicheng/moge-2-vitl) · [Poids relief](https://huggingface.co/Ruicheng/moge-2-vitl-normal)

## Résultat Colab L4

Les deux profils ont passé deux inférences par photo, sur une image en
384 × 512 et sa version miroir. Avec les poids officiels et PyTorch CUDA
FP32, le RMSE des normales est au plus 2,049 × 10⁻⁶ et l’écart angulaire
moyen au plus 0,00444°.
