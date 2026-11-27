# Depth Anything 3 Base

Le graphe porte le profil de LocalMesh avec quatre vues et une référence fixée
à la première vue. Il applique l’attention locale puis alternée entre les
vues avec les mêmes jetons caméra, normalisations et rotations. Il calcule
la profondeur et la confiance, les six composantes des rayons et leur
confiance, ainsi que les neuf valeurs de la tête de pose directe. Le fork
utilise les deux prédictions pour ses règles de latéralité. Le solveur
géométrique existant transforme les rayons en extrinsèques sur CPU.
[Architecture DA3](https://github.com/ByteDance-Seed/Depth-Anything-3)

Compiler `//libraries/depth_anything_v3:da3_probe`. La conversion lit les
poids et le descripteur du bundle LocalMesh. Elle extrait 331 des 437 tenseurs
FP32, qui couvrent la voie avec référence fixée et sans extrinsèques d’entrée.

```bash
python -B libraries/depth_anything_v3/prepare.py --probe "$probe" \
  --config "$descriptor" --checkpoint "$checkpoint" --output "$converted"
python -B libraries/depth_anything_v3/validate.py --probe "$probe" \
  --weights "$converted" --runtime "$runtime" --upstream "$localmesh" \
  --photos "$photos" --scratch "$scratch" --report "$report" \
  --expected-device cuda:0 --expected-pjrt cuda
```

Le probe prend `BACKEND WEIGHTS H W INPUT OUTPUT REPEATS`. Les entrées
`[4,3,H,W]` sont normalisées par le préprocesseur LocalMesh. Les côtés
sont divisibles par 14. Le fichier de sortie FP32 little-endian concatène
`[4,H,W,2]`, `[4,8H/14,8W/14,7]`, puis `[4,9]`. Les résultats utilisent
l’ordre front, right, back, left.
[Bundle testé](https://huggingface.co/Qtn-Cls/LocalMeshEngine/tree/72e67cbb4e8442b9257ed5a239f4662c776c3a5d/multivue/cameras)

Les quatre photos du samouraï ont été comparées à 504 × 504 avec
PyTorch 2.8 CUDA, en FP32 sur Colab L4. Le probe a vérifié le placement
PJRT `cuda` sur `cuda:0`. Le seuil est de 0,001 RMSE.

| Sortie | RMSE |
| --- | ---: |
| Profondeur | 2,428 × 10⁻⁷ |
| Confiance de profondeur | 1,411 × 10⁻⁵ |
| Rayons | 1,631 × 10⁻⁷ |
| Confiance des rayons | 5,241 × 10⁻⁴ |
| Tête de pose directe | 2,164 × 10⁻⁷ |
