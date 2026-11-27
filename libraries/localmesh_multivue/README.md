# Portage de LocalMesh

Référence fonctionnelle de LocalMesh Engine 1.2.0, commit
32987f5966330f70dac19182be1d92ad760a8506. Le bundle public vérifié est
Qtn-Cls/LocalMeshEngine, révision 72e67cbb4e8442b9257ed5a239f4662c776c3a5d.
[Code de référence](https://github.com/Quentincls/localmesh-engine/tree/32987f5966330f70dac19182be1d92ad760a8506) · [Bundle public](https://huggingface.co/Qtn-Cls/LocalMeshEngine/tree/72e67cbb4e8442b9257ed5a239f4662c776c3a5d)

| Utilisation | Poids chargés par LocalMesh | État du port |
| --- | --- | --- |
| Détourage | BiRefNet_HR | [Graphe validé à 1 024 × 1 024 sur L4](../birefnet/README.md) |
| Conditionnement des images | DINOv3 ViT-L/16 | [Encodeur natif](../dinov3/README.md) |
| Caméras et latéralité des quatre vues | Depth Anything 3 DA3-BASE | [Deux têtes validées à 504 × 504 sur quatre vues](../depth_anything_v3/README.md) |
| Traits de haute résolution des quatre vues | NAF champ.safetensors | [Encodeur et échantillonnage natifs](../naf/README.md) |
| Occupation, une image | TRELLIS.2 ss_flow_img_dit_1_3B_64_fp8 | [Graphe du débruiteur](../trellis2/README.md) |
| Occupation, quatre images | Pixal3D structure_mv_fp8 | [Graphe du débruiteur projeté](../trellis2/README.md) |
| Occupation d’une grille 64³ | TRELLIS ss_dec_conv3d_16l8_fp16 | [Décodeur TRELLIS réutilisé](../trellis/README.md) |
| Forme 512 et 1024, une image | TRELLIS.2 slat_flow_img2shape | Prédiction de vitesse validée pour les deux checkpoints sur L4 |
| Forme 512 et 1024, quatre images | Pixal3D forme_512_mv_fp8 et forme_1024_mv_fp8 | Prédiction de vitesse validée pour les deux checkpoints sur L4 |
| Texture 512 et 1024 | TRELLIS.2 slat_flow_imgshape2tex | Prédiction de vitesse validée pour les deux checkpoints sur L4 |
| Décodage et réencodage de forme | shape_dec_next_dc_f16c32_fp8 et shape_enc_next_dc_f16c32_fp8 | [VAE de forme validés sur L4](../trellis2_vae/README.md) |
| Décodage et encodage PBR | tex_dec_next_dc_f16c32_fp8 et tex_enc_next_dc_f16c32_fp8 | [VAE PBR validés sur L4](../trellis2_vae/README.md) |
| Focale et ajustement facultatif du relief | MoGe‑2 ViT-L et ViT-L Normal | [Deux profils validés sur L4](../moge2/README.md) |
| Recette Pixal3D `extreme` | Quatre débruiteurs BF16 et trois décodeurs FP16 officiels | [Débruiteurs et projection](../trellis2/README.md) · [VAE sparse](../trellis2_vae/README.md) |

Les recettes une photo et quatre vues réutilisent le même décodeur
d’occupation, le décodeur de forme et la texturation. La projection de photos,
le contourage par grille duale et la finition GLB sont des opérations de
pipeline avec leurs propres dépendances. Le fork charge les probes ZML
par une configuration locale pour les modèles des voies mono-photo et
quatre vues. Le moteur conserve ses opérations de fusion, de géométrie,
de projection et d’écriture GLB. [Fork LocalMesh](https://github.com/MrMybal/localmesh-engine_zml)

Pour les tests, télécharger les poids publics dans un dossier temporaire et
conserver les poids DINOv3 dans la bibliothèque locale de l’utilisateur.
La licence Meta doit être obtenue avant de télécharger ce modèle.
[Modèle DINOv3 officiel](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)

## Validation de l’intégration

Sur Colab L4, le fork a lancé DINOv3 et NAF sur les quatre vues de l’exemple
samouraï et le décodeur d’occupation sur une grille 16³. Le décodeur a
donné un RMSE de 4,974 × 10⁻⁵ face à PyTorch sur le premier essai de
comparaison des logits. La validation Windows couvre le raccordement avec
des probes simulés et des tenseurs CPU.

La recette d’occupation en 12 pas donne un RMSE final de 0,0005443 pour
une photo et de 0,00002096 pour quatre photos face à PyTorch CUDA en FP32.
Dans le second essai, les deux sorties comptent les mêmes 3 763 cellules
occupées. Les flux sparse de forme et de texture peuvent être configurés
dans le fork par checkpoint. Le débruiteur de forme 1024 a aussi été testé
sur 20 000 lignes sparse avec 4 101 jetons de conditionnement.

Les six profils de forme et de texture ont passé deux pas avec
l’échantillonneur du fork sur 48 cellules. Le RMSE maximal au deuxième pas
est de 0,0002363. Les comparaisons des nouveaux graphes figurent dans
leurs fiches de portage.

La CLI du fork a produit des GLB texturés du samouraï au palier `draft`,
graine 42, sur Colab L4. Le test rouvre les fichiers pour vérifier la
géométrie, les UV et la texture. Le rapport contrôle le placement de
chaque étape ZML jusqu’au décodeur de texture.

| Entrée | Faces du GLB | Étapes ZML vérifiées | Pic mémoire de la carte |
| --- | ---: | ---: | ---: |
| Une photo | 79 672 | 9 | 19,93 Gio |
| Quatre vues | 74 817 | 12 | 20,18 Gio |

La recette Pixal3D `extreme` a produit un GLB texturé de 460 300 faces
sur la L4 via `Engine.generate`. L’essai utilise les quatre passes de
24 pas par défaut, une grille de forme à 1 536 et les poids TencentARC
officiels. Le rapport contrôle onze étapes ZML, de BiRefNet_HR et MoGe-2
jusqu’aux deux décodeurs FP16 Pixal3D.

## Exécution sans CUDA

Le fork LocalMesh accepte `cpu`, `rocm`, `vulkan` et `metal` dans sa
configuration ZML. Ces modes exécutent les réseaux sur le périphérique
sélectionné et réservent au CPU l’échantillonnage, le contourage sparse,
le dépliage UV et la cuisson du GLB. L’export portable utilise le volume
PBR décodé. Le mode `--topology quad` reconstruit le maillage sur CPU,
transfère les normales de la surface dense avec Embree et livre le GLB
texturé accompagné d’un OBJ en quads.

Sous Windows, le palier draft mono-photo sur `cpu:0` a produit un GLB
texturé avec une carte de normales et un OBJ de 37 856 quads avec les vrais
poids. Les tests avec sorties simulées couvrent aussi les deux fusions à
quatre vues et Pixal3D.

Sur l’iGPU AMD Raphael `gfx1036`, la projection de patches DINOv3 ViT-L/16
en FP32 a passé deux inférences de 32 × 32 avec les vrais poids. La sonde
vérifie le périphérique propriétaire de la sortie avant la copie vers
l’hôte. Les 4 096 valeurs de chaque sortie sont comparées à une
convolution de référence sur CPU, avec une erreur absolue maximale de
`1e-4` autorisée.

| PJRT | Erreur absolue maximale | RMSE | Validation Windows |
|---|---:|---:|---|
| `iree_rocm` | `2.62e-6` | `3.12e-7` | BiRefNet-HR à 1 024 × 1 024 validé avec les deux réglages HIP ci-dessous. La chaîne complète attend une nouvelle validation. |
| `iree_vulkan` | `3.51e-6` | `3.12e-7` | Le graphe précédent demandait 2,11 Gio en une allocation. La projection revue et la chaîne complète attendent une validation Vulkan. |

Depuis un PowerShell configuré pour `clang-cl`, `lld-link` et le SDK
Windows, lancer les deux sondes en séquence. Le script fonctionne avec
un seul cœur logique à la priorité minimale, écrit son journal dans
`validation-output` et nettoie le dossier temporaire du cas avant de quitter.

```powershell
$model = 'D:\chemin\dinov3-vitl16-pretrain-lvd1689m\model.safetensors'
$scratch = 'D:\_Project\zml_lib_Temp'
$amdIcd = 'C:\chemin\amd-vulkan64.json'
python -B tests/windows/validate_localmesh_amd_projection.py rocm --checkpoint $model --scratch $scratch
python -B tests/windows/validate_localmesh_amd_projection.py vulkan --checkpoint $model --scratch $scratch --amd-icd $amdIcd
```

La sonde BiRefNet-HR en ROCm à 1 024 × 1 024 reproduit les 400 539 pixels
de premier plan de ZML CPU au seuil de 0,8, avec une RMSE de 4,19 × 10⁻⁷.
Ce résultat porte sur le détourage, avec les poids officiels et les deux
variables HIP dans l’environnement du processus. L’essai complet sur
l’iGPU reste à faire.

```powershell
$env:AMD_OPT_FLUSH = '0'
$env:AMD_SERIALIZE_KERNEL = '3'
$assets = 'D:\_Project\zml_lib_Temp\localmesh-native'
python -B tests/windows/validate_localmesh_portable.py --assets $assets --backend rocm --matte-size 1024
```

## Reconstruction du binaire

Construire ces cibles avec les options de la plateforme indiquées dans la
documentation de build du dépôt. La CLI LocalMesh charge les exécutables
et les poids préparés depuis le fichier JSON passé dans `--zml-config`.

```bash
bazel build \
  //libraries/birefnet:birefnet_probe \
  //libraries/depth_anything_v3:da3_probe \
  //libraries/dinov3:dinov3_condition \
  //libraries/moge2:moge2_probe \
  //libraries/naf:naf_probe \
  //libraries/trellis:trellis_structure \
  //libraries/trellis2:trellis2_flow \
  //libraries/trellis2_vae:trellis2_vae_probe
```

Le [guide du fork](https://github.com/MrMybal/localmesh-engine_zml/blob/main/docs/ZML.md)
décrit les clés de poids, les chemins temporaires et le test GLB. Sous
Windows, sélectionner PJRT `iree_cuda` pour l’exécution native CUDA.
Le placement L4 a été vérifié sur Linux avec le PJRT `cuda`.

```powershell
$env:ZML_TEMP_ROOT = 'D:\_Project\zml_lib_Temp'
$targets = @(
    'birefnet:birefnet_probe', 'depth_anything_v3:da3_probe',
    'dinov3:dinov3_condition', 'moge2:moge2_probe', 'naf:naf_probe',
    'trellis:trellis_structure', 'trellis2:trellis2_flow',
    'trellis2_vae:trellis2_vae_probe'
)
$selection = foreach ($target in $targets) { '--target'; "//libraries/$target" }
python .\scripts\build.py --engine ..\zml_win @selection
```
