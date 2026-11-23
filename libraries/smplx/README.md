# SMPL-X neutre en ZML

Ce portage Zig/ZML évalue le maillage SMPL-X au profil utilisé par GVHMR/SuperMotion. Il prend 10 coefficients de forme, 10 d’expression, les rotations axe-angle du corps, des yeux et de la mâchoire, 12 coefficients PCA par main et une translation XYZ. Le graphe applique les corrections de forme et de pose, la cinématique des 55 articulations et le skinning linéaire de 10 475 sommets. Le modèle source fournit les 20 908 triangles dans leur ordre d’origine. [Référence SMPL-X](https://github.com/vchoutas/smplx)

Les articulations retournées correspondent aux 55 entrées de l’arbre SMPL-X. Les coefficients de mains laissés à zéro donnent la pose moyenne du modèle. Le convertisseur lit les tableaux numériques du `SMPLX_NEUTRAL.npz` local, vérifie la topologie et exporte huit tenseurs FP32. Les poids SMPL-X sont fournis par l’utilisateur, conformément aux conditions de leur licence. [Licence SMPL-X de référence](https://github.com/vchoutas/smplx/blob/master/LICENSE)

## Formats locaux

`smplx_io.py <SMPLX_NEUTRAL.npz> <nouveau.safetensors>` prépare les poids. Le lanceur Zig accepte `smplx_mesh.exe <backend> <poids.safetensors> <nombre_de_lignes> <entree.f32> <sortie.f32> <repetitions>`. Les deux buffers sont contigus et stockés en FP32 little-endian sans en-tête. Chaque ligne d’entrée contient, dans l’ordre, `betas[10]`, `expression[10]`, `global_orient[3]`, `body_pose[63]`, `jaw_pose[3]`, `leye_pose[3]`, `reye_pose[3]`, `left_hand_pose[12]`, `right_hand_pose[12]` et `transl[3]`. La sortie associe à chaque ligne ses sommets XYZ puis ses articulations cinématiques XYZ. Le lanceur réutilise les poids et le graphe compilé par lots de 16 lignes. Le dernier lot est complété par des zéros et seules ses lignes utiles sont exportées.

## Validation Windows CUDA

Depuis la racine du dépôt, reprendre la configuration `$runtime`, `$py`, `$upstream`, `$env:ZML_TEMP_ROOT`, les DLL CUDA et le tableau `$cuda` du rejeu GVHMR. Le test emploie le checkpoint neutre local et 17 lignes de paramètres, avec des déformations du corps, des mains et du visage. Il vérifie aussi la translation, les rotations proches de zéro et de π, le lot incomplet, la réutilisation du modèle et le périphérique propriétaire de chaque sortie. [Configuration CUDA et rejeu vidéo](../gvhmr/README.md#vidéo-à-caméra-mobile-vers-smpl-x)

Sous Windows x64, les quatre lots ont été exécutés sur la RTX 3090 via `iree_cuda` avec un seul chargement des poids. Face à PyTorch FP32, les écarts absolus maximaux sont de `5,9604645e-7` pour les sommets et de `4,1723251e-7` pour les articulations. La topologie est identique à celle de la référence.

```powershell
$checkpoint = (Join-Path $upstream 'inputs\checkpoints\body_models\smplx\SMPLX_NEUTRAL.npz')
& $py -B .\scripts\build.py --target //libraries/smplx:smplx_mesh `
  -- --jobs=12 --host_compilation_mode=fastbuild @cuda --then $py -B .\libraries\smplx\validate.py `
  --probe '{bin}\libraries\smplx\smplx_mesh.exe' --checkpoint $checkpoint `
  --backend cuda --expected-device 'RTX 3090'
```
