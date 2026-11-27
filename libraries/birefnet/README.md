# BiRefNet_HR

Le graphe calcule le matte alpha de BiRefNet_HR avec Swin V1-L aux deux
échelles, le décodeur multirésolution et les convolutions déformables.
Il utilise les poids FP16 officiels et effectue les calculs en FP32.
Le fork applique l’alpha à la photo avant le recadrage.
[Checkpoint et code officiels](https://huggingface.co/ZhengPeng7/BiRefNet_HR/tree/a7a562f6fd16021180f2f4348f4de003a2d3d1e1)

Compiler `//libraries/birefnet:birefnet_probe`. La conversion du
`model.safetensors` local vérifie le checkpoint et prépare 675 tenseurs,
y compris les biais relatifs des fenêtres Swin.

```bash
python -B libraries/birefnet/prepare.py --probe "$probe" \
  --checkpoint "$original/model.safetensors" --output "$converted"
python -B libraries/birefnet/validate.py --probe "$probe" \
  --weights "$converted" --original "$original" --image "$photo" \
  --scratch "$scratch" --report "$report" --device cuda:0 --pjrt cuda
```

Entrée FP32 little-endian `[1,3,H,W]` en RGB normalisée ImageNet.
Sortie `[1,1,H,W]` de probabilités alpha. Les côtés sont des multiples de
64 de 64 à 1 024. Le fork utilise 1 024 × 1 024.

La dernière convolution 1 × 1 projette les 192 canaux du décodeur avant
le redimensionnement bilinéaire. La branche image projette séparément
ses 48 canaux à pleine résolution. Les deux projections et le biais
donnent les logits. Ce calcul évite un tenseur intermédiaire de 240
canaux à la résolution de sortie.

Sous Windows, le pilote de validation LocalMesh accepte
`--backend rocm --matte-size 1024` ou
`--backend vulkan --matte-size 1024`. Il compare le masque au même
graphe sur ZML CPU. Ajouter `--logits` pour comparer les valeurs avant
le sigmoid. [Pilote de validation Windows](https://github.com/MrMybal/localmesh-engine_zml/blob/main/tests/windows/validate_localmesh_portable.py)

Sur Colab L4 à 1 024 × 1 024 avec le graphe initial et les poids officiels, le RMSE face à
PyTorch CUDA en FP32 est de 2,472 × 10⁻⁷. Le masque binaire au seuil de
0,8 a une intersection sur union de 1,0.
