# VAE sparse de TRELLIS.2

Les quatre profils partagent les convolutions submanifold, les blocs
ConvNeXt, la normalisation et les transitions S2C et C2S. Les poids
matriciels des checkpoints LocalMesh restent en E4M3FN et les graphes
calculent en FP32. L’appelant gère le changement de coordonnées entre
les graphes compilés et les subdivisions transmises à la texture.
Les sorties de forme alimentent le contourage de grille duale du fork.
[TRELLIS.2](https://github.com/microsoft/TRELLIS.2) · [Poids LocalMesh](https://huggingface.co/Qtn-Cls/LocalMeshEngine/tree/72e67cbb4e8442b9257ed5a239f4662c776c3a5d/TRELLIS.2-4B/ckpts_fp8)

| Profil | Famille de checkpoint |
| --- | --- |
| shape_encode | shape_enc_next_dc_f16c32_fp8 |
| shape_decode | shape_dec_next_dc_f16c32_fp8 |
| texture_encode | tex_enc_next_dc_f16c32_fp8 |
| texture_decode | tex_dec_next_dc_f16c32_fp8 |

Le convertisseur accepte également les deux décodeurs FP16 Pixal3D avec
les mêmes profils `shape_decode` et `texture_decode`. Comparer cette
variante avec `validate.py --pixal-weights-root` et les étapes
`pixal_shape_decode` et `pixal_texture_decode` dans la configuration du fork.

Compiler `//libraries/trellis2_vae:trellis2_vae_probe`. L’outil prépare
un dossier par checkpoint avec neuf fichiers d’étage et un manifeste.

```bash
python -B libraries/trellis2_vae/prepare.py --probe "$probe" --profile shape_decode \
  --checkpoint "$weights" --config "$descriptor" --output "$prepared_dir"
python -B libraries/trellis2_vae/validate.py --config "$native_config" \
  --fork "$fork" --weights-root "$original_checkpoints" --report "$report"
```

Le dossier préparé est le champ `weights` de l’étape dans le fork. Les
coordonnées suivent `[batch,x,y,z]`. Les poids de convolution utilisent
l’ordre d’origine `[C_out,K_d,K_h,K_w,C_in]`. Chaque appel reçoit les
traits et les 27 indices entiers de voisinage par ligne, encodés en FP32.
Un indice égal au nombre de lignes donne la valeur zéro. Les encodeurs
renvoient la moyenne et la log-variance avant l’échantillonnage.
Le décodeur de forme prédit ses subdivisions. Le décodeur de texture
utilise les subdivisions de forme ou celles mémorisées par le réencodage.

## Résultat Colab L4

Les quatre profils ont été comparés aux VAE PyTorch du fork en FP32.
La référence calcule les convolutions à l’aide de torch.nn.functional.conv3d
sur une grille dense remplie de zéros hors des cellules actives. Les
subdivisions et le support final sont identiques sur la fixture testée.

| Profil | RMSE finale | Lignes de sortie |
| --- | ---: | ---: |
| shape_encode | 1,568 × 10⁻⁶ | 1 |
| texture_encode | 2,133 × 10⁻⁶ | 1 |
| shape_decode | 2,051 × 10⁻⁵ | 181 |
| texture_decode | 6,847 × 10⁻⁷ | 181 |

Avec les décodeurs FP16 Pixal3D évalués en FP32 sur la même fixture,
le décodeur de forme donne un RMSE de 1,971 × 10⁻⁵ et le décodeur PBR
de 6,687 × 10⁻⁷. Le support est identique entre les moteurs aux quatre
subdivisions et sur les 183 lignes finales.

La roue FlexGEMM de Colab a donné un écart de 0,00637 sur la tête de forme
sur la première comparaison des logits. La comparaison FP32 sur grille dense
vérifie séparément la convolution et la topologie.
