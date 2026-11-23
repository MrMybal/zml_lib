# BoT-SORT

Le graphe ZML calcule en FP32 la distance IoU entre les pistes prédites et les détections XYXY, avec la fusion facultative du score de détection. Avec Ultralytics 8.3.253, l’adaptateur GVHMR utilise ce graphe pour l’association principale, la confirmation des pistes, l’association des détections à faible score et la comparaison des pistes actives et perdues pour repérer les doublons. Ultralytics applique sur l’hôte le filtre de Kalman, la compensation du mouvement de caméra, le solveur LAPJV et les règles de suppression des doublons. [BoT-SORT](https://github.com/ultralytics/ultralytics/blob/v8.3.253/ultralytics/trackers/bot_sort.py#L187) · [BYTETracker](https://github.com/ultralytics/ultralytics/blob/v8.3.253/ultralytics/trackers/byte_tracker.py) · [Calcul des coûts](https://github.com/ultralytics/ultralytics/blob/v8.3.253/ultralytics/trackers/utils/matching.py#L57)

`video.py --yolov8-probe … --botsort-cost-probe …` conserve un processus et une session ZML pendant le suivi d’une vidéo. Le graphe est compilé au premier appel et quand la capacité augmente. Chaque axe est arrondi à la puissance de deux supérieure avec un minimum de huit. L’adaptateur reçoit les grandes matrices par blocs et reconstitue la matrice dans l’ordre attendu par LAPJV. Les boîtes orientées et les coûts d’association avec ReID utilisent le calcul Ultralytics. Les appels directs à l’IoU des associations secondaires et des doublons sont liés à la classe de suivi utilisée pour la vidéo. Pour les autres versions d’Ultralytics, l’option ZML couvre les appels à `get_dists`. `--verify` compare chaque matrice ZML à celle d’Ultralytics. [Rejeu GVHMR](../gvhmr/README.md#rejeu-complet-à-caméra-mobile)

## Protocole du lanceur

`botsort_cost.exe cuda` traite une suite de requêtes binaires sur stdin, en little-endian. Chaque requête contient trois entiers de 32 bits `T`, `D` et `fusion`, les coordonnées FP32 `[T,4]` des pistes, puis les détections FP32 `[D,5]` dans l’ordre `x1,y1,x2,y2,score`. `fusion` vaut 0 ou 1. Le lanceur accepte de 1 à 16 384 pistes et de 1 à 512 détections par requête. L’adaptateur Python renvoie directement les matrices vides lorsque l’un des deux axes est vide.

Le processus renvoie sur stdout une matrice FP32 `[T,D]` en ordre ligne, immédiatement après chaque requête. Le client possède le buffer qu’il reçoit. Les diagnostics de compilation et de placement sont écrits sur stderr. Fermer stdin termine la session et libère ses ressources. Le graphe ne charge aucun poids.

## Validation CUDA sous Windows x64

Depuis la racine du dépôt, avec les DLL du fork `zml_win` et un Python contenant NumPy et SciPy.

```powershell
$py = 'D:\ComfyUI\.venv\Scripts\python.exe'
$tools = (Resolve-Path '..\zml_win\bazel-win-tools').Path
$env:ZML_TEMP_ROOT = 'D:\_Project\zml_lib_Temp'
$env:CUDA_VISIBLE_DEVICES = '0'
$env:CUDA_CACHE_DISABLE = '1'
$env:ZML_CUDA_PJRT_DLL = Join-Path $tools 'iree-build\python\iree\_pjrt_libs\cuda\iree_pjrt_cuda_dylib.dll'
$env:IREE_PJRT_COMPILER_LIB_PATH = Join-Path $tools 'iree-wheel\iree\compiler\_mlir_libs\IREECompiler.dll'
$env:IREE_PJRT_IREE_COMPILER_OPTIONS = '--iree-cuda-target=sm_86'
$cuda = @('--@zml//platforms:cpu=false', '--@zml//platforms:cuda=true', '--@zml//platforms:rocm=false', '--@zml//platforms:vulkan=false')
& $py -B .\scripts\build.py --target //libraries/botsort:botsort_cost `
  -- --jobs=12 --host_compilation_mode=fastbuild @cuda --then $py -B .\libraries\botsort\validate_costs.py `
  --probe '{bin}\libraries\botsort\botsort_cost.exe' --backend cuda --expected-device 'RTX 3090'
```

Le test compare les matrices à une référence NumPy FP32 pour 195 jeux synthétiques et vérifie les affectations calculées par SciPy. Il contrôle le placement de chaque sortie avant sa copie sur l’hôte et vérifie le rejet d’une requête contenant un NaN dans un processus distinct. Chaque essai nettoie son espace de travail sous `D:\_Project\zml_lib_Temp`.

Ajouter `--ultralytics` à la commande avec un Python équipé d’Ultralytics 8.3.253 et de LAPX 0.10.0. Ce mode rejoue une séquence synthétique de 195 images avec le tracker Ultralytics d’origine et avec l’adaptateur ZML. Il compare les identifiants, les boîtes et les scores de toutes les pistes et vérifie le placement des coûts à chaque étape. Un cas isolé compare la suppression d’un doublon parmi 513 pistes perdues, ce qui force deux requêtes ZML.

Sur la RTX 3090, les 195 matrices FP32 sont identiques à NumPy, avec les mêmes affectations SciPy et 1 329 paires retenues au seuil de 0,8. Le rejeu à caméra mobile synthétique a retrouvé les 479 lignes de pistes d’Ultralytics. Il a vérifié 263 matrices ZML, dont 196 pour l’association principale et la confirmation, 36 pour les détections à faible score et 31 pour les doublons. Les deux blocs du cas de 513 pistes perdues ont produit le même dédoublonnage qu’Ultralytics. Le placement a été vérifié sur `iree_cuda` avant chaque copie. Le rejeu GVHMR jusqu’à SMPL-X reste à réaliser avec le runtime et les poids 4DAnyone de CyTools. [Mesures](port.json)
