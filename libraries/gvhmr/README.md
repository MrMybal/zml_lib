# GVHMR SIGA24

Le réseau temporel SIGA24 exécute 12 blocs RoPE et les trois têtes de sortie en Zig/ZML. Il traite la visibilité des articulations, les conditions de caméra et les caractéristiques HMR2. Pour une séquence de plus de 120 images, l’attention reproduit la fenêtre glissante du modèle. Le lanceur vidéo utilise le prétraitement CyTools et peut reconstruire le maillage SMPL-X en ZML avec le checkpoint local. [Référence GVHMR](https://github.com/zju3dv/GVHMR)

Le modèle reçoit une séquence continue de 1 à 4 096 images valides. Le fichier `conditions.npz` fournit quatre tableaux flottants, avec le même nombre `T` d’images.

| Tableau | Dimensions | Contenu |
| --- | --- | --- |
| `obs` | `[T,17,3]` | Points COCO dans le repère normalisé du recadrage et leurs scores |
| `f_cliffcam` | `[T,3]` | Décalages du centre de la boîte et taille, divisés par la focale |
| `f_cam_angvel` | `[T,6]` | Mouvement angulaire encodé en 6D et normalisé avec les statistiques SIGA24 |
| `f_imgseq` | `[T,1024]` | Caractéristiques d’image HMR2 |

La sortie `pred_x` contient les 151 composantes normalisées de mouvement. Elle regroupe les rotations 6D des 21 articulations, les 10 coefficients de forme moyennés sur la séquence, deux orientations racine 6D et la vitesse locale 3D. `pred_cam` contient les trois paramètres de caméra et `static_conf_logits` les six logits d’immobilité. Les buffers FP32 little-endian du lanceur occupent `[T,1084]` en entrée et `[T,160]` en sortie, dans l’ordre du tableau puis des trois sorties ci-dessus. Le client Zig détient les buffers et libère le modèle avant sa session.

## Validation Windows x64 avec le checkpoint local

Utiliser le Python de CyTools qui peut importer la référence GVHMR, avec PyTorch, NumPy et SafeTensors. `GVHMR_SOURCE` pointe sur le checkout de référence et `GVHMR_CHECKPOINT` sur `gvhmr_siga24_release.ckpt`. Le test compare les trois sorties à PyTorch avec les mêmes poids, à sept et à 137 images, et exécute deux inférences par modèle chargé. Ajouter `--conditions` pour rejouer une séquence de caractéristiques fournie par l’appelant.

Le checkpoint CyTools a fourni 247 tenseurs FP32. La comparaison sous Windows x64 sur `cpu:0` utilise des conditions synthétiques déterministes.

| Images | Écart maximal `pred_x` | Écart maximal `pred_cam` | Écart maximal `static_conf_logits` |
| ---: | ---: | ---: | ---: |
| 7 | `4,5776367e-5` | `2,3841858e-7` | `1,2159348e-5` |
| 137 | `6,4849854e-5` | `5,9604645e-7` | `1,2397766e-5` |

```powershell
$env:ZML_TEMP_ROOT = (Join-Path (Split-Path -Parent (Get-Location).Path) 'zml_lib_Temp')
$env:ZML_CPU_PJRT_DLL = (Resolve-Path '..\zml_win\bazel-win-tools\pjrt-cpu\pjrt_cpu.dll').Path
$py = (Get-Command python).Source
$cpu = @('--@zml//platforms:cuda=false', '--@zml//platforms:rocm=false', '--@zml//platforms:vulkan=false')
& $py -B .\scripts\build.py --target //libraries/gvhmr:gvhmr_temporal -- --jobs=12 --host_compilation_mode=fastbuild @cpu --then $py -B .\libraries\gvhmr\validate.py --probe '{bin}\libraries\gvhmr\gvhmr_temporal.exe' --upstream $env:GVHMR_SOURCE --checkpoint $env:GVHMR_CHECKPOINT
```

## Utiliser un fichier de conditions

Compiler sans `--then` pour disposer du lanceur dans `validation-output\bin`, puis convertir les poids et lancer une prédiction.

```powershell
$probe = '.\validation-output\bin\libraries\gvhmr\gvhmr_temporal.exe'
& $py -B .\libraries\gvhmr\temporal_io.py convert --probe $probe --checkpoint $env:GVHMR_CHECKPOINT --weights '.\validation-output\gvhmr-temporal.safetensors'
& $py -B .\libraries\gvhmr\temporal_io.py predict --probe $probe --weights '.\validation-output\gvhmr-temporal.safetensors' --conditions $env:GVHMR_CONDITIONS --output '.\validation-output\gvhmr-motion.npz' --backend cpu
```

Le test importe le checkout de référence fourni par l’utilisateur. Les poids, les ressources SMPL-X et le code amont sont fournis séparément et relèvent de leurs conditions respectives. [Licence amont de GVHMR](https://github.com/zju3dv/GVHMR/blob/6ec3ca39336c50492c0fae65fba2fb831fc7d866/LICENSE)

## Vidéo à caméra fixe vers SMPL-X

`video.py` utilise le checkout local de 4DAnyone pour décoder la vidéo. Il accepte des sondes ZML pour YOLOv8x, ViTPose-H, HMR2 et les étapes de reconstruction. En l’absence d’une sonde de prétraitement, le lanceur utilise le modèle correspondant du checkout GVHMR sous PyTorch/CUDA. Les poids GVHMR, HMR2, ViTPose, YOLOv8 et SMPL-X viennent de `inputs/checkpoints` dans ce checkout.

L’option `--verify` compare les trois sorties temporelles et les sommets SMPL-X à la référence PyTorch sur les mêmes caractéristiques vidéo. Elle relit aussi l’archive produite. Sans `--output`, cette archive est temporaire et supprimée après le test. Avec `--output`, le NPZ contient les paramètres SMPL-X, les sommets, les faces, les paramètres intrinsèques de la caméra, les points 2D et la chronologie.

La validation Windows x64 a utilisé les poids CyTools locaux et 12 images de `tennis.mp4`. Le prétraitement a tourné sur une RTX 3090 et le modèle temporel ZML sur `cpu:0`. Le NPZ relu contient 10 475 sommets par image et 20 908 triangles. Les écarts absolus maximaux avec la chaîne de référence PyTorch sont de `1,3411045e-6` pour les sommets monde et de `9,5367432e-7` pour les sommets caméra.

```powershell
$env:GVHMR_4DANYONE = '<checkout local 4DAnyone>'
$env:GVHMR_SOURCE = (Join-Path $env:GVHMR_4DANYONE 'third_party\GVHMR')
$video = (Join-Path $env:GVHMR_SOURCE 'docs\example_video\tennis.mp4')
& $py -B .\scripts\build.py --target //libraries/gvhmr:gvhmr_temporal -- --jobs=12 --host_compilation_mode=fastbuild @cpu --then $py -B .\libraries\gvhmr\video.py --probe '{bin}\libraries\gvhmr\gvhmr_temporal.exe' --four-d-anyone $env:GVHMR_4DANYONE --upstream $env:GVHMR_SOURCE --video $video --frames 12 --verify
```

### Réseau temporel CUDA sur RTX 3090

Sous Windows x64, `--verify` exécute deux inférences avec le réseau temporel déjà chargé. Chaque sortie doit appartenir au périphérique sélectionné avant la copie vers l’hôte. La comparaison PyTorch utilise les mêmes caractéristiques de la vidéo et le même checkpoint SIGA24. Le runtime CyTools calcule le post-traitement et les maillages SMPL-X sur l’hôte. [Guide CUDA de zml_win](https://github.com/MrMybal/zml_win/blob/windows-x64/docs/windows-x64.md#cuda-avec-iree-pjrt)

Les deux rejeux de `tennis.mp4` ont utilisé YOLOv8, ViTPose et HMR2 sur la RTX 3090. Les quatre sorties temporelles ZML ont confirmé `iree_cuda` et le même périphérique. Le cas long de 312 images exerce la fenêtre glissante de 120 images. Chaque NPZ relu contient 10 475 sommets par image et 20 908 triangles.

| Images canoniques | Images source distinctes | Écart maximal des sommets monde | Écart maximal des sommets caméra |
| ---: | ---: | ---: | ---: |
| 12 | 11 | `1,1622906e-6` | `9,5367432e-7` |
| 312 | 311 | `2,4065375e-6` | `2,2351742e-6` |

Les écarts sont absolus et comparés à PyTorch FP32. Le rééchantillonnage temporel par défaut utilise la cadence canonique de `29,827092` images par seconde du clip.

```powershell
$runtime = 'D:\_Project\_CyTools\CyTools_VideoToGaussian\runtime'
$py = (Join-Path $runtime '4danyone-python\Scripts\python.exe')
$fourD = (Join-Path $runtime 'sources\4DAnyone')
$upstream = (Join-Path $fourD 'third_party\GVHMR')
$video = (Join-Path $upstream 'docs\example_video\tennis.mp4')
$tools = (Resolve-Path '..\zml_win\bazel-win-tools').Path
$env:ZML_TEMP_ROOT = 'D:\_Project\zml_lib_Temp'
$env:CUDA_VISIBLE_DEVICES = '0'
$env:CUDA_CACHE_DISABLE = '1'
$env:ZML_CUDA_PJRT_DLL = (Join-Path $tools 'iree-build\python\iree\_pjrt_libs\cuda\iree_pjrt_cuda_dylib.dll')
$env:IREE_PJRT_COMPILER_LIB_PATH = (Join-Path $tools 'iree-wheel\iree\compiler\_mlir_libs\IREECompiler.dll')
$env:IREE_PJRT_IREE_COMPILER_OPTIONS = '--iree-cuda-target=sm_86'
$cuda = @('--@zml//platforms:cpu=false', '--@zml//platforms:cuda=true', '--@zml//platforms:rocm=false', '--@zml//platforms:vulkan=false')
& $py -B .\scripts\build.py --target //libraries/gvhmr:gvhmr_temporal `
  -- --jobs=12 --host_compilation_mode=fastbuild @cuda --then $py -B .\libraries\gvhmr\video.py `
  --probe '{bin}\libraries\gvhmr\gvhmr_temporal.exe' --four-d-anyone $fourD --upstream $upstream `
  --video $video --frames 312 --backend cuda --expected-device 'RTX 3090' --verify
```

Utiliser `--frames 12` pour retrouver l’extrait court. Chaque essai supprime ses fichiers intermédiaires et son NPZ de validation sous `D:\_Project\zml_lib_Temp`.

## Vidéo à caméra mobile vers SMPL-X

`--camera moving` ajoute l’odométrie visuelle SimpleVO de GVHMR au prétraitement CyTools. Le pipeline convertit les rotations monde vers caméra successives en signal angulaire 6D pour le réseau temporel et le décodeur corporel. `--odometry-probe` active le solveur ZML et `--sift-probe` lui fournit les correspondances calculées en ZML. En l’absence de ces options, le lanceur utilise le solveur SIFT et PyCOLMAP de GVHMR. [Démonstrateur GVHMR épinglé](https://github.com/zju3dv/GVHMR/blob/6ec3ca39336c50492c0fae65fba2fb831fc7d866/tools/demo/demo.py#L146)

La référence PyCOLMAP peut être installée dans le Python du lanceur ou chargée depuis CyTools avec `--pycolmap-site`. Le paquet local doit avoir la même ABI Python que le lanceur. L’archive du mode mobile ajoute `camera_rotations_w2c` (`[T,3,3]`) et `camera_angular_velocity_6d` (`[T,6]`). `--verify` compare les maillages monde et caméra avec PyTorch sur la même trajectoire et relit ces deux tableaux.

Les premiers essais Windows x64 ont utilisé le réseau temporel ZML et les maillages PyTorch sur l’hôte. Ils ont chargé les poids CyTools locaux, PyCOLMAP 3.12.6 et le clip à 15 images par seconde. Chaque essai a confirmé que la RTX 3090 possédait les deux sorties du réseau temporel via `iree_cuda`. Chaque archive relue contient 10 475 sommets par image et 20 908 triangles. Le rejeu de 12 images à caméra fixe a également réussi.

| Images distinctes | Rotation maximale depuis la première image | Écart absolu maximal des sommets monde | Écart absolu maximal des sommets caméra |
| ---: | ---: | ---: | ---: |
| 48 | 10,16° | `1,8328428e-5` | `1,9073486e-6` |
| 195 | 24,33° | `1,0704994e-4` | `2,3841858e-6` |

Les écarts sont mesurés face à PyTorch FP32.

Réutiliser la configuration ci-dessus et la vidéo locale `Tokyo-Walk_rgb.mp4`. Le cas de 195 images couvre la fenêtre temporelle glissante. Utiliser `--frames 48` pour rejouer l’extrait court.

### Réseau temporel et maillages SMPL-X en CUDA

`--smplx-probe` active le portage SMPL-X sur le même backend que le réseau temporel. Le portage charge le `SMPLX_NEUTRAL.npz` local de CyTools et calcule les maillages monde et caméra dans une seule session ZML. L’archive ajoute `global_kinematic_joints` et `incam_kinematic_joints` de dimensions `[T,55,3]`. `--verify` contrôle l’appartenance des sorties au GPU, compare le maillage à paramètres identiques, puis compare la chaîne complète à PyTorch FP32. [Portage et formats SMPL-X](../smplx/README.md)

Les essais Windows x64 de 48 et 195 images ont confirmé `iree_cuda` et la RTX 3090 pour les deux inférences temporelles et chaque lot de sommets et d’articulations. Le tableau donne les écarts absolus maximaux face à PyTorch FP32.

| Images distinctes | Lots SMPL-X sur deux passages | Sommets SMPL-X à paramètres identiques | Sommets monde de la chaîne complète | Sommets caméra de la chaîne complète |
| ---: | ---: | ---: | ---: | ---: |
| 48 | 12 | `9,5367432e-7` | `1,8328428e-5` | `1,9073486e-6` |
| 195 | 50 | `9,5367432e-7` | `1,0704994e-4` | `2,3841858e-6` |

Le rejeu de 12 images de `tennis.mp4` avec le même portage SMPL-X a également réussi en mode `--camera fixed`. L’écart absolu maximal des sommets monde était de `1,2814999e-6` face à PyTorch FP32.

### Corrections des contacts et cinématique inverse en CUDA

`--postprocess-probe` applique en ZML la correction de translation, la remise au sol et la cinématique inverse aux paramètres corporels décodés. Le graphe utilise le squelette SMPL-X de 22 articulations du checkpoint neutre local et les probabilités de contact des chevilles, des pieds et des poignets. En caméra mobile, il pondère les déplacements des contacts et lisse la translation horizontale. En caméra fixe, il applique la correction bornée de l’écart au repère caméra puis la moyenne des contacts actifs. Pour les deux modes, il reconstruit les cibles temporelles et exécute deux passes CCD pour chaque jambe et chaque bras. [Post-traitement de référence épinglé](https://github.com/zju3dv/GVHMR/blob/6ec3ca39336c50492c0fae65fba2fb831fc7d866/hmr4d/model/gvhmr/utils/postprocess.py)

La vérification compare la translation et les rotations corporelles axe-angle à entrées identiques. Elle compare ensuite les maillages monde et caméra de la chaîne complète à PyTorch FP32, contrôle le périphérique propriétaire des sorties du réseau temporel, du post-traitement et de SMPL-X, puis relit l’archive.

Les six essais Windows x64 sont passés sur la RTX 3090 via `iree_cuda` avec les poids locaux CyTools. Les trois essais isolés ont mesuré un écart maximal de `1,6987324e-6` radian sur les rotations. Les essais vidéo ont exécuté deux inférences temporelles et deux passes complètes du graphe de correction par chargement. Les écarts ci-dessous sont absolus et mesurés face à PyTorch FP32.

| Caméra | Images canoniques | Translation corrigée à entrées identiques | Rotation corrigée à entrées identiques (rad) | Sommets monde de la chaîne complète |
| --- | ---: | ---: | ---: | ---: |
| Mobile | 48 | `2,3841858e-7` | `2,1606684e-6` | `1,8179417e-5` |
| Mobile | 195 | `1,4305115e-6` | `4,7683716e-6` | `1,0681152e-4` |
| Fixe | 12 | `1,1920929e-7` | `8,4936619e-7` | `1,4156103e-6` |

Le clip mobile de 195 images a une rotation maximale estimée de `24,33°`. Chaque archive vidéo relue contient les corrections, 10 475 sommets et 55 articulations cinématiques par image dans les deux repères, ainsi que 20 908 triangles.

### Décodage des paramètres corporels en CUDA

`--decode-probe` effectue le décodage en ZML sur le même backend que le réseau temporel. Le lanceur exporte la moyenne et l’écart-type FP32 des 151 canaux depuis la configuration SIGA24 locale. Le graphe convertit les rotations 6D en axe-angle, produit les coefficients de forme et calcule la translation depuis la boîte et les paramètres intrinsèques de la caméra. Dans le repère monde, il lisse et compose les rotations de caméra, intègre la vitesse locale à partir de la première image et applique le changement de repère de GVHMR. [Décodeur de référence épinglé](https://github.com/zju3dv/GVHMR/blob/6ec3ca39336c50492c0fae65fba2fb831fc7d866/hmr4d/model/gvhmr/utils/endecoder.py#L161) · [Reconstruction du repère monde](https://github.com/zju3dv/GVHMR/blob/6ec3ca39336c50492c0fae65fba2fb831fc7d866/hmr4d/model/gvhmr/pipeline/gvhmr_pipeline.py#L322)

Le graphe calcule l’angle horizontal avec `atan2` et le produit vectoriel des directions. Ce calcul conserve la précision des petites rotations lorsque le produit scalaire FP32 est proche de 1. La validation compare les paramètres bruts à une référence PyTorch FP32 qui utilise cette formulation avec une composition séquentielle. Elle mesure aussi séparément les écarts avec le pipeline GVHMR amont, dont le calcul d’angle utilise `acos`.

Les orientations sont comparées sous forme de matrices pour tenir compte des représentations axe-angle équivalentes près de π. La chaîne vidéo vérifie les corrections, les maillages dans les deux repères et le périphérique propriétaire de chaque sortie native.

Les six essais Windows x64 ont réussi sur la RTX 3090 via `iree_cuda` avec les poids locaux CyTools. Les trois essais isolés et les trois séquences vidéo ont exécuté deux décodages par chargement. Le tableau donne les écarts absolus maximaux en FP32. Les deux premières mesures de décodage utilisent les mêmes sorties temporelles et la référence stabilisée.

| Caméra | Images canoniques | Matrice d’orientation monde du décodeur | Translation monde du décodeur | Sommets monde face à PyTorch stabilisé | Sommets monde face au pipeline amont |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mobile | 48 | `8,4936619e-7` | `2,7939677e-7` | `1,8477440e-6` | `3,6486983e-4` |
| Mobile | 195 | `1,7816055e-6` | `9,5367432e-7` | `6,6757202e-6` | `3,0812025e-3` |
| Fixe | 12 | `1,0430813e-6` | `2,6077032e-8` | `2,2500753e-6` | `2,2500753e-6` |

Sur les 195 images, la translation monde avant correction diffère de `3,1778812e-3` par rapport au décodeur amont. Le clip contient une rotation caméra estimée allant jusqu’à `24,33°`.

```powershell
$video = (Join-Path $runtime 'sources\Video-Depth-Anything\assets\example_videos\Tokyo-Walk_rgb.mp4')
$colmapSite = (Join-Path $runtime 'python\Lib\site-packages')
& $py -B .\scripts\build.py --target //libraries/gvhmr:gvhmr_temporal `
  --target //libraries/gvhmr:gvhmr_decode --target //libraries/gvhmr:gvhmr_postprocess --target //libraries/smplx:smplx_mesh `
  -- --jobs=12 --host_compilation_mode=fastbuild @cuda --then $py -B .\libraries\gvhmr\video.py `
  --probe '{bin}\libraries\gvhmr\gvhmr_temporal.exe' --four-d-anyone $fourD --upstream $upstream `
  --decode-probe '{bin}\libraries\gvhmr\gvhmr_decode.exe' `
  --postprocess-probe '{bin}\libraries\gvhmr\gvhmr_postprocess.exe' --smplx-probe '{bin}\libraries\smplx\smplx_mesh.exe' `
  --video $video --frames 195 --camera moving --pycolmap-site $colmapSite `
  --backend cuda --expected-device 'RTX 3090' --verify
```

Le test des corrections seules couvre une image mobile ainsi que des séquences de 27 images en caméra mobile et fixe. Avec la même configuration CUDA, utiliser `--case moving1`, `--case moving27` ou `--case fixed27`.

```powershell
& $py -B .\scripts\build.py --target //libraries/gvhmr:gvhmr_postprocess `
  -- --jobs=12 --host_compilation_mode=fastbuild @cuda --then $py -B .\libraries\gvhmr\validate_postprocess.py `
  --probe '{bin}\libraries\gvhmr\gvhmr_postprocess.exe' --four-d-anyone $fourD --upstream $upstream `
  --case moving27 --backend cuda --expected-device 'RTX 3090'
```

Pour vérifier le décodeur seul, utiliser `--case single`, `--case fixed137` ou `--case moving137`.

```powershell
& $py -B .\scripts\build.py --target //libraries/gvhmr:gvhmr_decode `
  -- --jobs=12 --host_compilation_mode=fastbuild @cuda --then $py -B .\libraries\gvhmr\validate_decode.py `
  --probe '{bin}\libraries\gvhmr\gvhmr_decode.exe' --four-d-anyone $fourD --upstream $upstream `
  --case moving137 --backend cuda --expected-device 'RTX 3090'
```

### Odométrie SimpleVO en CUDA

`--odometry-probe` prend des correspondances entre images à demi-résolution, avec une image clé toutes les huit images et la dernière image du clip. Il utilise SIFT et FLANN d’OpenCV sur l’hôte en l’absence de `--sift-probe`. Le graphe de géométrie ZML estime en FP64 les homographies et les matrices essentielles à huit points par RANSAC, affine leurs coefficients, puis choisit la pose relative à partir des points triangulés. Le graphe de trajectoire compose les poses monde vers caméra en FP32, interpole les rotations et les translations, puis produit le signal 6D utilisé par GVHMR. L’archive ajoute `camera_translation_unscaled_w2c` (`[T,3]`), composée de déplacements non métriques dont l’échelle dépend de chaque paire d’images. [SimpleVO de référence](https://github.com/zju3dv/GVHMR/blob/6ec3ca39336c50492c0fae65fba2fb831fc7d866/hmr4d/utils/preproc/relpose/simple_vo.py) · [Sélection de la géométrie dans COLMAP 3.12.6](https://github.com/colmap/colmap/blob/3.12.6/src/colmap/estimators/two_view_geometry.cc#L404)

Avec `--verify`, le lanceur répète les calculs d’odométrie avec les deux graphes déjà compilés, vérifie le périphérique propriétaire de chaque sortie et compare la trajectoire ZML à une composition FP64 indépendante. Il compare aussi la trajectoire aux estimations PyCOLMAP calculées sur les mêmes correspondances. Le solveur déterministe ZML et le RANSAC de PyCOLMAP peuvent sélectionner des poses relatives différentes. La comparaison finale du maillage avec PyTorch emploie le même signal de caméra ZML dans les deux chaînes.

Les extraits de `Tokyo-Walk_rgb.mp4` à 15 images/s ont été validés sous Windows x64 avec les poids locaux CyTools. La géométrie SimpleVO, le réseau temporel, le décodage corporel, les corrections de contact, la cinématique inverse et SMPL-X ont utilisé `iree_cuda` sur la RTX 3090. Chaque archive relue contient 10 475 sommets par image et 20 908 triangles.

| Images | Paires résolues par homographie | Sorties CUDA d’odométrie sur deux passages | Écart angulaire de trajectoire avec PyCOLMAP, max. / RMS | Écart maximal des sommets monde avec PyTorch |
| ---: | ---: | ---: | ---: | ---: |
| 48 | 6 / 6 | 14 | `0,696°` / `0,444°` | `1,79e-6` |
| 195 | 25 / 25 | 52 | `3,236°` / `1,780°` | `1,24e-5` |

À poses relatives identiques, l’écart maximal du signal 6D face à la composition FP64 indépendante est de `3,10e-7` sur les 195 images. La rotation maximale estimée pour ce clip depuis la première image est de `25,26°`. [Mesures complètes](port.json)

Le rejeu de 12 images de `tennis.mp4` en mode fixe a également passé la vérification du réseau temporel jusqu’à SMPL-X, avec un écart maximal de `2,25e-6` sur les sommets monde.

Avec les variables Windows et CUDA définies ci-dessus, exécuter la chaîne sur 195 images ou passer `--frames 48` pour le clip court. Chaque commande place les fichiers de compilation et d’essai sous `D:\_Project\zml_lib_Temp` et les supprime à la sortie.

```powershell
$video = (Join-Path $runtime 'sources\Video-Depth-Anything\assets\example_videos\Tokyo-Walk_rgb.mp4')
$colmapSite = (Join-Path $runtime 'python\Lib\site-packages')
& $py -B .\scripts\build.py --target //libraries/gvhmr:gvhmr_simplevo --target //libraries/gvhmr:gvhmr_temporal `
  --target //libraries/gvhmr:gvhmr_decode --target //libraries/gvhmr:gvhmr_postprocess --target //libraries/smplx:smplx_mesh `
  -- --jobs=12 --host_compilation_mode=fastbuild @cuda --then $py -B .\libraries\gvhmr\video.py `
  --four-d-anyone $fourD --upstream $upstream --video $video --frames 195 --camera moving --pycolmap-site $colmapSite `
  --odometry-probe '{bin}\libraries\gvhmr\gvhmr_simplevo.exe' --probe '{bin}\libraries\gvhmr\gvhmr_temporal.exe' `
  --decode-probe '{bin}\libraries\gvhmr\gvhmr_decode.exe' --postprocess-probe '{bin}\libraries\gvhmr\gvhmr_postprocess.exe' `
  --smplx-probe '{bin}\libraries\smplx\smplx_mesh.exe' --backend cuda --expected-device 'RTX 3090' --verify
```

Pour vérifier la géométrie sur des poses connues avec des correspondances aberrantes, utiliser `--case panoramic` ou `--case essential`.

Sur la RTX 3090, les deux cas ont conservé au moins 162 correspondances sur 192 par paire. L’écart angulaire maximal face aux poses connues est de `0,00102°` en rotation pure et de `0,07424°` avec translation. Le cosinus entre les directions de translation estimée et connue est d’au moins `0,999978`.

```powershell
& $py -B .\scripts\build.py --target //libraries/gvhmr:gvhmr_simplevo `
  -- --jobs=12 --host_compilation_mode=fastbuild @cuda --then $py -B .\libraries\gvhmr\validate_simplevo.py `
  --probe '{bin}\libraries\gvhmr\gvhmr_simplevo.exe' --case panoramic --backend cuda --expected-device 'RTX 3090'
```

### Extraction SIFT et appariement en CUDA

`--sift-probe` envoie les images clés RGB8 décodées à leur résolution d’origine au lanceur `gvhmr_sift.exe`. ZML applique le même filtre bicubique entier que pour ViTPose-H et HMR2. Les pixels réduits alimentent SIFT sur le même périphérique. En FP32, ZML convertit les couleurs en niveaux de gris, construit les octaves gaussiennes, localise les extrema des différences de gaussiennes et élimine les points de faible contraste et les réponses de bord. Il calcule jusqu’à deux orientations par point, puis un descripteur SIFT normalisé de 128 composantes à partir d’un échantillonnage bilinéaire de la grille orientée. Le graphe conserve au plus 4 096 descripteurs par image. À la résolution de travail, les images doivent avoir entre 10 et 4 096 pixels par côté et au plus 2 097 152 pixels. [Prétraitement vidéo ZML](#prétraitement-vidéo-par-zml) · [Article SIFT](https://doi.org/10.1023%2FB%3AVISI.0000029664.99615.94) · [Extraction de référence OpenCV](https://github.com/opencv/opencv/blob/4.12.0/modules/features2d/src/sift.dispatch.cpp)

Le graphe d’appariement évalue les distances L2 au carré et applique le seuil de Lowe de 0,7 sur les deux plus proches voisins exacts. Le lanceur amont utilise un index FLANN approché. Le lanceur vidéo sélectionne les images clés du clip canonique déjà décodé. Il calcule la calibration SimpleVO à la résolution réduite et transmet les correspondances au solveur. `--verify` rejoue la réduction, SIFT et l’appariement deux fois, contrôle le périphérique de chaque sortie, compare les valeurs RGB8 réduites à PyAV et compare la trajectoire aux références PyCOLMAP sur les mêmes correspondances et OpenCV/PyCOLMAP sur les mêmes images. [Appariement de référence GVHMR](https://github.com/zju3dv/GVHMR/blob/6ec3ca39336c50492c0fae65fba2fb831fc7d866/hmr4d/utils/preproc/relpose/model/cv2_matcher.py)

```text
gvhmr_sift.exe half cuda hauteur largeur images trames.rgb correspondances.f32 repetitions [--reduced-rgb trames_reduites.rgb] [descripteurs.f32]
gvhmr_sift.exe cuda hauteur largeur images trames_deja_reduites.rgb correspondances.f32 repetitions [descripteurs.f32]
```

Les trames d’entrée sont des tableaux contigus RGB8 `[N,H,W,3]`. En mode `half`, `--reduced-rgb` enregistre les trames réduites du dernier passage pour la comparaison avec PyAV. Sur RTX 3090, `validate_sift.py --half` a retrouvé exactement les pixels de PyAV, les descripteurs et les correspondances SIFT sur huit trames de dimensions paires et impaires. Les correspondances sont aussi identiques dans l’essai qui laisse les trames réduites sur le GPU. [Résultats des essais isolés](port.json)

Les deux extraits de `Tokyo-Walk_rgb.mp4` ont été validés sous Windows x64 avec les poids CyTools locaux sur la RTX 3090 via `iree_cuda`. Les maillages SMPL-X relus contiennent 10 475 sommets par image et 20 908 triangles. Le tableau consigne le premier portage SIFT, avec les graines des références laissées à leur état courant.

| Images | Correspondances ZML / OpenCV | Sorties SIFT contrôlées en CUDA | Écart angulaire maximal avec OpenCV/PyCOLMAP | Écart maximal des sommets monde avec PyTorch à signal caméra identique |
| ---: | ---: | ---: | ---: | ---: |
| 48 | 8 676 / 8 766 | 544 | `3,127°` | `2,15e-6` |
| 195 | 34 100 / 34 547 | 2 026 | `13,590°` | `9,54e-6` |

Sur les 195 images, la comparaison avec PyCOLMAP à correspondances ZML identiques donne un écart maximal de `2,342°` par paire et de `6,919°` sur la trajectoire. La comparaison des sommets utilise le signal caméra ZML dans les deux chaînes. [Mesures détaillées](port.json)

Utiliser `--frames 48` pour le rejeu court. Comme pour les essais précédents, les commandes emploient `D:\_Project\zml_lib_Temp` et suppriment les fichiers temporaires à la sortie.

```powershell
$video = (Join-Path $runtime 'sources\Video-Depth-Anything\assets\example_videos\Tokyo-Walk_rgb.mp4')
$colmapSite = (Join-Path $runtime 'python\Lib\site-packages')
& $py -B .\scripts\build.py --target //libraries/gvhmr:gvhmr_sift --target //libraries/gvhmr:gvhmr_simplevo `
  --target //libraries/gvhmr:gvhmr_temporal --target //libraries/gvhmr:gvhmr_decode `
  --target //libraries/gvhmr:gvhmr_postprocess --target //libraries/smplx:smplx_mesh `
  -- --jobs=12 --host_compilation_mode=fastbuild @cuda --then $py -B .\libraries\gvhmr\video.py `
  --four-d-anyone $fourD --upstream $upstream --video $video --frames 195 --camera moving --pycolmap-site $colmapSite `
  --sift-probe '{bin}\libraries\gvhmr\gvhmr_sift.exe' --odometry-probe '{bin}\libraries\gvhmr\gvhmr_simplevo.exe' `
  --probe '{bin}\libraries\gvhmr\gvhmr_temporal.exe' --decode-probe '{bin}\libraries\gvhmr\gvhmr_decode.exe' `
  --postprocess-probe '{bin}\libraries\gvhmr\gvhmr_postprocess.exe' --smplx-probe '{bin}\libraries\smplx\smplx_mesh.exe' `
  --backend cuda --expected-device 'RTX 3090' --verify
```

Le test isolé vérifie les descripteurs et compare le résultat du seuil de Lowe à SciPy FP64. Il mesure aussi les erreurs de reprojection sur une translation et sur une rotation avec changement d’échelle. Une image uniforme vérifie le cas sans correspondance.

Sur la RTX 3090, le test a vérifié 975 correspondances avec SciPy FP64. La translation a produit 444 correspondances, dont 99,10 % à moins de deux pixels, avec une erreur médiane de 0,121 pixel. La rotation de 18° et le changement d’échelle ont produit 531 correspondances, dont 98,87 % à moins de deux pixels, avec une erreur médiane de 0,179 pixel. L’image uniforme a produit zéro descripteur.

```powershell
& $py -B .\scripts\build.py --target //libraries/gvhmr:gvhmr_sift `
  -- --jobs=12 --host_compilation_mode=fastbuild @cuda --then $py -B .\libraries\gvhmr\validate_sift.py `
  --probe '{bin}\libraries\gvhmr\gvhmr_sift.exe' --backend cuda --expected-device 'RTX 3090'
```

Ajouter `--half` à cette commande pour vérifier le redimensionnement RGB8 et la transmission des trames réduites à SIFT sur le périphérique.

Le mode `--video` de `validate_sift.py` rejoue SimpleVO avec PyAV, NumPy et SciPy. Il utilise les premières images décodées dans l’ordre du fichier et sélectionne les images clés de SimpleVO. Un Python local équipé de ces trois modules suffit au test. Pour reproduire le test de 195 images depuis la racine du dépôt sous Windows x64, utiliser les deux sondes ZML et un clip fourni par l’appelant.

```powershell
$py = 'D:\ComfyUI\.venv\Scripts\python.exe'
$video = 'D:\_Project\_CyTools\CyTools_VideoStudio\runtime\engines\stereo\runtime\source\video-depth-anything\assets\example_videos\Tokyo-Walk_rgb.mp4'
$tools = (Resolve-Path '..\zml_win\bazel-win-tools').Path
$env:ZML_TEMP_ROOT = 'D:\_Project\zml_lib_Temp'
$env:CUDA_VISIBLE_DEVICES = '0'
$env:CUDA_CACHE_DISABLE = '1'
$env:ZML_CUDA_PJRT_DLL = Join-Path $tools 'iree-build\python\iree\_pjrt_libs\cuda\iree_pjrt_cuda_dylib.dll'
$env:IREE_PJRT_COMPILER_LIB_PATH = Join-Path $tools 'iree-wheel\iree\compiler\_mlir_libs\IREECompiler.dll'
$env:IREE_PJRT_IREE_COMPILER_OPTIONS = '--iree-cuda-target=sm_86'
$cuda = @('--@zml//platforms:cpu=false', '--@zml//platforms:cuda=true', '--@zml//platforms:rocm=false', '--@zml//platforms:vulkan=false')
& $py -B .\scripts\build.py --target //libraries/gvhmr:gvhmr_sift --target //libraries/gvhmr:gvhmr_simplevo `
  -- --jobs=12 --host_compilation_mode=fastbuild @cuda --then $py -B .\libraries\gvhmr\validate_sift.py `
  --probe '{bin}\libraries\gvhmr\gvhmr_sift.exe' --odometry-probe '{bin}\libraries\gvhmr\gvhmr_simplevo.exe' `
  --video $video --frames 195 --backend cuda --expected-device 'RTX 3090'
```

Le script compare les pixels réduits à PyAV, puis vérifie les descripteurs et les correspondances SIFT sur ces mêmes pixels. Il compare la trajectoire ZML à SciPy FP64 sur les mêmes transformations relatives. Il écrit le rapport JSON sur stdout, les diagnostics sur stderr et supprime ses fichiers sous `D:\_Project\zml_lib_Temp` à sa sortie.

Sur une copie locale de `Tokyo-Walk_rgb.mp4`, la validation indépendante sous Windows x64 a traité les 26 images clés du clip de 195 images sur RTX 3090 avec `iree_cuda`. Les 17 971 200 valeurs RGB8 sont identiques à PyAV 18.1.0. Les descripteurs et les 34 100 correspondances sont identiques à ceux obtenus en fournissant à SIFT les images réduites par PyAV. SimpleVO a interpolé les 195 poses de caméra et a résolu les 25 paires par homographie. Avec les mêmes poses relatives, l’écart maximal des coefficients de rotation face à SciPy FP64 est de `2,05e-7`. Le rejeu jusqu’à SMPL-X de ce portage est à reprendre après restauration du runtime CyTools VideoToGaussian, devenu indisponible pendant la comparaison HMR2. [Calibration GVHMR](https://github.com/zju3dv/GVHMR/blob/6ec3ca39336c50492c0fae65fba2fb831fc7d866/hmr4d/utils/preproc/relpose/utils.py#L160) · [Mesures](port.json)

### Raffinement des homographies SimpleVO

Le solveur ZML évalue 128 hypothèses d’homographie. Il affine le meilleur modèle de chacun des quatre groupes pendant huit itérations, en gardant les améliorations du consensus, puis retient le meilleur résultat. Le choix de la pose plane triangule les correspondances qui soutiennent cette homographie. Les 96 tirages de la matrice essentielle gardent leur séquence. [Raffinement local de référence dans COLMAP](https://github.com/colmap/colmap/blob/3.12.6/src/colmap/optim/loransac.h#L145)

La vérification fixe la graine de FLANN à 0 puis réinitialise PyCOLMAP à 0 avant chacune des deux trajectoires de référence. Sur les mêmes correspondances OpenCV, six exécutions successives de PyCOLMAP sans cette réinitialisation ont différé de jusqu’à `9,319°`. L’écart de `13,590°` du premier portage venait d’un rejeu où la comparaison des correspondances ZML avait déjà avancé cet état aléatoire. Le tableau mesure les deux versions du solveur avec les mêmes références fixées.

| Référence de trajectoire sur 195 images | Ancien solveur, max. / RMS | Solveur corrigé, max. / RMS |
| --- | ---: | ---: |
| OpenCV et PyCOLMAP sur les mêmes images | `6,144°` / `3,801°` | `2,896°` / `1,980°` |
| PyCOLMAP sur les mêmes correspondances SIFT ZML | `6,919°` / `4,448°` | `2,710°` / `1,595°` |

Le rejeu des 195 images avec les poids CyTools a vérifié 2 134 sorties sur la RTX 3090 via `iree_cuda`, dont 52 pour SimpleVO sur deux passages. Les 25 paires ont pris la branche homographie, avec au moins 1 062 correspondances retenues par paire. Avec le même signal caméra ZML dans les deux chaînes, l’écart maximal des sommets monde avec PyTorch est de `1,43e-5`. Les essais géométriques `panoramic` et `essential` ont aussi réussi. [Mesures détaillées](port.json)

La commande PowerShell de la section [Extraction SIFT et appariement en CUDA](#extraction-sift-et-appariement-en-cuda) rejoue cette validation avec `--frames 195 --verify`.

### Prétraitement vidéo par ZML

`--yolov8-probe` prépare les trames BGR8 par letterbox et exécute YOLOv8x COCO dans le même graphe ZML. La NMS et la reprojection sur le même périphérique fournissent à BoT-SORT les boîtes personne en pixels de la vidéo. Avec Ultralytics 8.3.253, `--botsort-cost-probe` confie à une session ZML persistante les coûts IoU des associations et du dédoublonnage. Ultralytics gère le solveur d’affectation et l’état du suivi sur l’hôte. Le seuil YOLOv8 de 0,5 laisse les détections dans la plage de l’association principale avec la configuration actuelle de BoT-SORT. Le test synthétique de BoT-SORT couvre l’association des scores plus faibles. [YOLOv8](../yolov8/README.md) · [BoT-SORT](../botsort/README.md)

`--crop-probe` réduit les images à demi-résolution, calcule les recadrages normalisés dans le même graphe ZML et partage le résultat avec ViTPose-H et HMR2. `--vitpose-probe` calcule les cartes de pose et les coordonnées articulaires UDP dans le même graphe ZML. `--hmr2-probe` calcule les jetons d’apparence dans ZML. [ViTPose-H](../vitpose/README.md) · [HMR2](../hmr2/README.md)

La mise à demi-résolution reçoit les trames RGB8 décodées du clip canonique. Elle reproduit le filtre bicubique entier de libswscale utilisé par GVHMR, y compris le calcul de la chrominance par paires en largeur paire. Chaque dimension est divisée par deux, arrondie à l’entier inférieur et bornée à un pixel. Le mode normal conserve les images réduites dans le graphe. `--verify` lit aussi cette sortie RGB8 pour la comparer pixel par pixel au filtre PyAV de la référence. [Lecture vidéo GVHMR](https://github.com/zju3dv/GVHMR/blob/6ec3ca39336c50492c0fae65fba2fb831fc7d866/hmr4d/utils/video_io_utils.py) · [Filtre bicubique libswscale](https://github.com/FFmpeg/FFmpeg/blob/n8.0/libswscale/utils.c)

Le graphe de recadrage résout l’affinité en FP64 à partir de trois points source arrondis en FP32. Il applique le filtre anti-crénelage gaussien (5 × 5) lorsque la boîte dépasse le seuil GVHMR, puis échantillonne les 192 colonnes centrales d’un recadrage carré de 256 pixels. L’interpolation bilinéaire utilise des fractions au 1/32 de pixel et un remplissage noir hors image. La sortie est un tenseur RGB FP32 normalisé selon ImageNet. En l’absence de `--crop-probe`, le lanceur utilise le prétraitement du checkout GVHMR. [Prétraitement GVHMR](https://github.com/zju3dv/GVHMR/blob/6ec3ca39336c50492c0fae65fba2fb831fc7d866/hmr4d/utils/preproc/vitfeat_extractor.py) · [Référence OpenCV](https://github.com/opencv/opencv/blob/4.14.0/modules/imgproc/src/imgwarp.cpp)

`gvhmr_crop.exe` désactive dans son processus la conversion automatique de FP64 vers FP32 par IREE. La précision de la grille aux arrondis à mi-pas est nécessaire pour retrouver les mêmes pixels qu’OpenCV. La normalisation et la sortie utilisent FP32.

```text
gvhmr_crop.exe half cuda hauteur largeur images lot trames.rgb boites.f32 recadrages.f32 repetitions [trames_reduites.rgb]
gvhmr_crop.exe cuda hauteur largeur images lot trames.rgb boites.f32 recadrages.f32 repetitions
```

Les trames sont des tableaux contigus RGB8 `[N,H,W,3]` à la résolution d’origine en mode `half` et à demi-résolution avec la seconde commande. Les boîtes FP32 little-endian `[N,3]` utilisent le repère vidéo d’origine `[centre_x, centre_y, côté]`. Les recadrages FP32 little-endian `[N,3,256,192]` et les trames RGB8 réduites optionnelles sont écrits dans de nouveaux fichiers.

Sur RTX 3090, `validate_half.py` a reproduit exactement les pixels RGB8 de PyAV et les recadrages d’OpenCV/PyTorch sur neuf images synthétiques. Les essais couvrent la résolution du clip, les dimensions impaires, une trame d’un pixel de haut, le filtre anti-crénelage, un lot incomplet et le mode qui renvoie uniquement les recadrages. `validate_crop.py` vérifie aussi le mode qui reçoit des images déjà réduites, les positions aux bords et l’arrondi à mi-pas.

```powershell
& $py -B .\scripts\build.py --target //libraries/gvhmr:gvhmr_crop `
  -- --jobs=12 --host_compilation_mode=fastbuild @cuda --then $py -B .\libraries\gvhmr\validate_half.py `
  --probe '{bin}\libraries\gvhmr\gvhmr_crop.exe' --upstream $upstream --backend cuda --expected-device 'RTX 3090'
```

Avec `--verify`, chaque sonde de prétraitement traite la séquence deux fois dans la même session. Le lanceur contrôle la provenance CUDA de chaque sortie, puis compare les trois modèles aux références PyTorch avec les mêmes recadrages et poids FP32. Le décodeur UDP est aussi comparé à OpenCV sur les cartes ZML. Les scores des 3 315 articulations de ce rejeu sont identiques. La comparaison du maillage utilise les mêmes observations, jetons d’apparence et signaux de caméra dans les deux chaînes.

Le rejeu du graphe partagé par ViTPose-H et HMR2, réalisé avant l’ajout de la réduction native dans SimpleVO, a traité les 195 images de `Tokyo-Walk_rgb.mp4` à 15 images/s sous Windows x64 avec les poids CyTools et `iree_cuda` sur RTX 3090. La réduction de 1 280 × 720 à 640 × 360 a donné les mêmes 134 784 000 valeurs RGB8 que PyAV. Les 195 recadrages FP32 concordent exactement avec la référence. Sur deux passages, les 98 lots du graphe commun ont chacun produit deux sorties contrôlées. La validation a contrôlé 2 916 sorties CUDA pour toute la chaîne et relu l’archive SMPL-X, qui contient 10 475 sommets par image et 20 908 triangles. [Mesures et révisions](port.json)

| Sortie | Écart absolu maximal |
| --- | ---: |
| YOLOv8x, coordonnées brutes à la résolution réseau | `0,00769 px` |
| YOLOv8x, scores | `1,04e-4` |
| Images RGB8 à demi-résolution face à PyAV | `0` |
| Recadrages normalisés face à OpenCV/PyTorch | `0` |
| ViTPose-H, cartes de chaleur | `1,91e-6` |
| ViTPose-H, coordonnées articulaires face à OpenCV sur les mêmes cartes | `0,000122 px` |
| ViTPose-H, coordonnées articulaires face à PyTorch et OpenCV | `0,000122 px` |
| HMR2, jetons d’apparence | `3,55e-6` |
| SMPL-X, sommets monde face à PyTorch avec le calcul d’angle stabilisé | `1,10e-5` |
| SMPL-X, sommets monde face au pipeline GVHMR amont | `5,92e-3` |

L’écart maximal de trajectoire avec OpenCV/PyCOLMAP reste à `2,90°`, avec 25 paires résolues par homographie. L’écart du maillage monde avec le pipeline amont inclut la différence entre les deux formulations de l’angle horizontal. [Décodage des paramètres corporels](#décodage-des-paramètres-corporels-en-cuda)

### Conditions du réseau temporel en ZML

`--conditions-probe` prépare les entrées du réseau SIGA24. Il exprime les coordonnées des 17 points COCO dans le repère de la boîte, conserve leur score sur les bords et annule le score hors boîte. Il calcule `f_cliffcam` à partir du centre optique, de la focale et de la taille de la boîte. Il applique les statistiques angulaires du pipeline lorsque `normalize_cam_angvel` est activé. Les jetons HMR2 complètent ces entrées. `--verify` compare les conditions ZML aux fonctions de référence GVHMR avant leur envoi au réseau. [Fonctions de géométrie GVHMR](https://github.com/zju3dv/GVHMR/blob/main/hmr4d/utils/geo/hmr_cam.py) · [Pipeline SIGA24](https://github.com/zju3dv/GVHMR/blob/main/hmr4d/model/gvhmr/pipeline/gvhmr_pipeline.py)

```text
gvhmr_conditions.exe cuda images entree.f32 stats.f32 sortie.f32 repetitions
```

L’entrée FP32 little-endian `[T,63]` contient 17 triplets `[x,y,score]`, la boîte `[centre_x,centre_y,côté]`, la caméra `[focale,centre_optique_x,centre_optique_y]` et six composantes angulaires. Le fichier `stats.f32` contient les six moyennes suivies des six écarts-types positifs. Une moyenne nulle et un écart-type de un conservent le signal angulaire. La sortie FP32 `[T,60]` regroupe les 17 triplets normalisés, les trois valeurs `f_cliffcam` et les six valeurs `f_cam_angvel`. La sonde accepte de 1 à 4 096 images.

Sous Windows x64 sur RTX 3090 via `iree_cuda`, les séquences synthétiques de 1, 195 et 4 096 images ont donné un écart maximal de `0` face à PyTorch FP32 pour les trois sorties. Le test couvre les bords inclus, les points immédiatement hors boîte, les scores signés, les caméras fixe et mobile et deux inférences dans la même session pour chaque longueur. Utiliser le Python local avec NumPy et PyTorch et reprendre les variables CUDA du test indépendant. [Configuration CUDA](#extraction-sift-et-appariement-en-cuda) · [Mesures](port.json)

```powershell
& $py -B .\scripts\build.py --target //libraries/gvhmr:gvhmr_conditions `
  -- --jobs=12 --host_compilation_mode=fastbuild @cuda --then $py -B .\libraries\gvhmr\validate_conditions.py `
  --probe '{bin}\libraries\gvhmr\gvhmr_conditions.exe' --backend cuda --expected-device 'RTX 3090'
```

### Rejeu complet à caméra mobile

Remplacer `--frames 195` par `--frames 12` pour un rejeu court. Les commandes utilisent `D:\_Project\zml_lib_Temp`. Le lanceur nettoie les données d’essai et le script de build nettoie ses fichiers temporaires à sa sortie.

Pour une installation CyTools VideoStudio, le moteur 4DAnyone utilise `runtime\engines\gaussian\runtime` dans VideoStudio. Ce dossier fournit le Python de `$py` et les sources de `$fourD` et `$upstream` lorsque le moteur est installé. Le clip `Tokyo-Walk_rgb.mp4` se trouve sous `runtime\engines\stereo\runtime\source\video-depth-anything\assets\example_videos`. Fournir le dossier de PyCOLMAP de la même version de Python via `--pycolmap-site` si nécessaire.

```powershell
$video = Join-Path $runtime 'sources\Video-Depth-Anything\assets\example_videos\Tokyo-Walk_rgb.mp4'
$colmapSite = Join-Path $runtime 'python\Lib\site-packages'
& $py -B .\scripts\build.py --target //libraries/yolov8:yolov8_probe --target //libraries/botsort:botsort_cost `
  --target //libraries/vitpose:vitpose_probe `
  --target //libraries/hmr2:hmr2_probe --target //libraries/gvhmr:gvhmr_crop `
  --target //libraries/gvhmr:gvhmr_sift --target //libraries/gvhmr:gvhmr_simplevo `
  --target //libraries/gvhmr:gvhmr_conditions --target //libraries/gvhmr:gvhmr_temporal --target //libraries/gvhmr:gvhmr_decode `
  --target //libraries/gvhmr:gvhmr_postprocess --target //libraries/smplx:smplx_mesh `
  -- --jobs=12 --host_compilation_mode=fastbuild @cuda --then $py -B .\libraries\gvhmr\video.py `
  --four-d-anyone $fourD --upstream $upstream --video $video --frames 195 --camera moving --pycolmap-site $colmapSite `
  --yolov8-probe '{bin}\libraries\yolov8\yolov8_probe.exe' --vitpose-probe '{bin}\libraries\vitpose\vitpose_probe.exe' `
  --botsort-cost-probe '{bin}\libraries\botsort\botsort_cost.exe' `
  --hmr2-probe '{bin}\libraries\hmr2\hmr2_probe.exe' --crop-probe '{bin}\libraries\gvhmr\gvhmr_crop.exe' `
  --sift-probe '{bin}\libraries\gvhmr\gvhmr_sift.exe' `
  --conditions-probe '{bin}\libraries\gvhmr\gvhmr_conditions.exe' `
  --odometry-probe '{bin}\libraries\gvhmr\gvhmr_simplevo.exe' --probe '{bin}\libraries\gvhmr\gvhmr_temporal.exe' `
  --decode-probe '{bin}\libraries\gvhmr\gvhmr_decode.exe' --postprocess-probe '{bin}\libraries\gvhmr\gvhmr_postprocess.exe' `
  --smplx-probe '{bin}\libraries\smplx\smplx_mesh.exe' --backend cuda --expected-device 'RTX 3090' --verify
```
