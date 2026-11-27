# NAF pour LocalMesh

Le port couvre les deux appels de la voie quatre vues de LocalMesh. Le
premier charge les 37 tenseurs FP32 de champ.safetensors et produit
les cartes de requêtes NAF. Le second calcule les traits aux coordonnées
demandées à partir de la carte des requêtes et du plan DINOv3 basse
résolution. Il applique l’attention à quatre têtes sur un voisinage 9 × 9
décalé aux bords, puis interpole les quatre pixels voisins.

Les photos préparées à 512 ou 1024 donnent une carte de 512 × 512 à
256 canaux dans la voie quatre vues. La texture Pixal3D accepte aussi
une carte pleine à 1024 × 1024. Le probe traite les coordonnées par groupes de 128 et réutilise
les deux cartes sur le GPU pendant cet appel. Le calcul utilise FP32.
[NAF valeo.ai](https://github.com/valeoai/NAF) · [Checkpoint distribué avec LocalMesh](https://huggingface.co/Qtn-Cls/LocalMeshEngine/tree/72e67cbb4e8442b9257ed5a239f4662c776c3a5d/multivue)

Compiler //libraries/naf:naf_probe. Les fichiers de tenseurs contiennent des
flottants FP32 little-endian contigus. Les deux cartes sont en NCHW.
Les coordonnées sont [N,2] dans l’ordre x,y normalisé, les résultats [N,C].

```bash
naf_probe encode cuda champ.safetensors 512 512 image.f32 queries.f32 2
naf_probe encode-half cuda champ.safetensors 1024 1024 image.f32 queries.f32 2
naf_probe sample cuda 512 512 32 32 1024 1024 queries.f32 features.f32 coords.f32 output.f32
python -B libraries/naf/validate.py \
  --probe "$probe" --upstream "$localmesh_checkout" --weights "$champ" \
  --scratch "$scratch" --image "$photo" --backend cuda \
  --expected-pjrt cuda --expected-device cuda:0 --reference-device cuda:0 --report "$report"
```

La comparaison exécute l’encodeur et la fonction d’échantillonnage PyTorch
du commit LocalMesh référencé dans le [suivi du portage](../localmesh_multivue/README.md).

## Validation avec les poids de LocalMesh

Colab L4, CUDA PJRT, calcul FP32. L’encodage et l’échantillonnage ont été
comparés à la référence CUDA avec le checkpoint champ.safetensors et
131 points par essai, dont des points aux bords de l’image.

| Taille de la photo | RMSE encodage | RMSE des traits échantillonnés |
| --- | ---: | ---: |
| 512 × 512 | 1,148 × 10⁻⁶ | 1,463 × 10⁻⁶ |
| 1 024 × 1 024 | 7,470 × 10⁻⁶ | 9,571 × 10⁻⁷ |

L’essai `--full-resolution` sur la photo du samouraï a donné un
RMSE de 7,712 × 10⁻⁶ avec une carte de 1 024 × 1 024. Sur les 131 points
échantillonnés, le RMSE des traits est de 1,120 × 10⁻⁶.
