# TRELLIS.2 et Pixal3D

Ce dossier porte le débruiteur à 30 blocs des checkpoints utilisés par
LocalMesh. Le même graphe accepte les flux d’occupation, de forme et de
texture, avec ou sans conditionnement projeté selon le profil. Il reçoit
des lignes de latents, des coordonnées 3D ordonnées z,y,x, un instant
d’échantillonnage, le conditionnement global et les traits projetés.
Le stockage conserve les matrices E4M3FN de LocalMesh. Le convertisseur
accepte aussi les checkpoints BF16 officiels de Pixal3D. Les deux familles
utilisent les valeurs d’origine pour les calculs en FP32.

Le probe accepte jusqu’à 4 096 lignes pour l’occupation, 65 536 lignes
pour les profils sparse et 8 192 jetons de conditionnement. Au-delà de
4 096 lignes, le graphe traite les requêtes d’attention par tuiles.
Le mode stream garde les poids chargés et le graphe compilé pendant chaque
étape d’échantillonnage. Le fork LocalMesh appelle ce mode pour les flux
d’occupation, de forme et de texture. Les [VAE de forme et PBR](../trellis2_vae/README.md) chargent leurs propres checkpoints.
Le décodeur d’occupation réutilise
[le décodeur TRELLIS v1](../trellis/README.md).
[Microsoft TRELLIS.2](https://github.com/microsoft/TRELLIS.2) · [Tencent Pixal3D](https://huggingface.co/TencentARC/Pixal3D)

Compiler //libraries/trellis2:trellis2_flow. Le convertisseur accepte le
descripteur et le SafeTensors locaux de la même variante. Il vérifie la
structure des blocs et prépare uniquement les valeurs utilisées par le graphe.

```bash
python -B libraries/trellis2/prepare.py --probe "$probe" --profile structure \
  --config "$descriptor" --checkpoint "$weights" --output "$converted"
python -B libraries/trellis2/validate_flow.py --probe "$probe" --profile structure \
  --original "$weights" --prepared "$converted" --scratch "$scratch" --report "$report" \
  --expected-device cuda:0 --expected-pjrt cuda --reference-device cuda:0
```

Le test compare tous les blocs avec les poids officiels sur des entrées
déterministes. Le profil structure_multiview lit les poids Pixal3D et
teste aussi les projections de vues. Les profils shape, shape_multiview,
texture et texture_multiview partagent le graphe et exigent leurs propres
descripteurs et poids.

## Validation avec les checkpoints de LocalMesh

Colab L4, PyTorch CUDA, calcul FP32 depuis les poids E4M3FN, seuil
0,001 RMSE. Les profils de forme et de texture ont été testés sur
48 lignes et 37 jetons de contexte. Les deux profils d’occupation ont
aussi été testés sur les 4 096 cellules de la grille 16³.

| Profil | RMSE, 48 lignes | RMSE, 4 096 lignes |
| --- | ---: | ---: |
| Occupation TRELLIS.2 | 1,205 × 10⁻⁵ | 2,111 × 10⁻⁶ |
| Occupation Pixal3D | 4,165 × 10⁻⁴ | 1,817 × 10⁻⁶ |
| Forme TRELLIS.2 512 | 8,258 × 10⁻⁶ | — |
| Forme TRELLIS.2 1024 | 6,373 × 10⁻⁶ | — |
| Forme Pixal3D 512 | 5,688 × 10⁻⁶ | — |
| Forme Pixal3D 1024 | 6,110 × 10⁻⁶ | — |
| Texture TRELLIS.2 512 | 6,258 × 10⁻⁶ | — |
| Texture TRELLIS.2 1024 | 9,160 × 10⁻⁶ | — |

La grille d’occupation complète utilise 1 029 jetons pour TRELLIS.2,
et cinq jetons globaux et des traits projetés pour Pixal3D.

Dans le test de forme 1024, 8 193 lignes sparse donnent un RMSE de
3,816 × 10⁻⁵. Le même checkpoint à 20 000 lignes et 4 101 jetons de
conditionnement donne un RMSE de 6,090 × 10⁻⁵.

## Checkpoints officiels de Pixal3D

Le test FP32 sur Colab L4 utilise 48 lignes et cinq jetons globaux. Il
compare les poids BF16 officiels de la recette Pixal3D à projection
mono-photo. [Révision des poids](https://huggingface.co/TencentARC/Pixal3D/tree/b0cb2e1b794cab9aa0ac38a95d794a4d9337437f)

| Étape | Profil | RMSE |
| --- | --- | ---: |
| Occupation | structure_multiview | 1,326 × 10⁻⁷ |
| Forme 512 | shape_multiview | 4,142 × 10⁻⁶ |
| Forme 1024 | shape_multiview | 5,807 × 10⁻⁶ |
| Texture 1024 | texture_multiview | 3,017 × 10⁻⁶ |

Le mode par flux binaire se lance avec
trellis2_flow PROFILE BACKEND PREPARED ROWS CONTEXT_ROWS stream.
Le probe écrit ready suivi d’un saut de ligne une fois la compilation
terminée. Chaque requête contient un octet zéro, puis en FP32 little-endian
les latents, les coordonnées, l’instant, le contexte et les traits projetés.
Utiliser une colonne de zéros pour les profils sans traits projetés.
La réponse contient une ligne de placement PJRT, puis les valeurs FP32.
La fermeture de stdin libère la session.
