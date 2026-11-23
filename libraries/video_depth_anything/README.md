# Video Depth Anything Small

Le modèle réutilise [DINOv2 ViT-S/14](../dinov2/README.md) et la tête DPT. Quatre modules d’attention temporelle relient jusqu’à 32 images d’un clip. `Model.infer` reçoit RGB FP32 normalisé `[T,3,H,W]` et produit une profondeur relative `[T,H,W]`. `H` et `W` sont des multiples de 14.

Le checkpoint local de CyTools VideoToStereo3D contient 351 tenseurs FP32. Sous Windows x64 sur `cpu:0`, l’erreur absolue maximale contre PyTorch est de `5.96046448e-6` sur un clip de deux images 28 × 42 et un clip de trois images 42 × 28.

    $env:ZML_TEMP_ROOT = 'D:\_Project\zml_lib_Temp'
    $stereo = (Join-Path $env:CYTOOLS_ROOT 'CyTools_VideoStudio\runtime\engines\stereo')
    $py = Join-Path $stereo 'runtime\engine\Scripts\python.exe'
    $env:ZML_CPU_PJRT_DLL = (Resolve-Path '..\zml_win\bazel-win-tools\pjrt-cpu\pjrt_cpu.dll').Path
    python .\scripts\build.py --target //libraries/video_depth_anything:video_depth_probe -- --jobs=8 --then $py -B .\libraries\video_depth_anything\validate.py --reference "$stereo\runtime\source\video-depth-anything" --checkpoint "$stereo\data\models\depth-vda-small\video_depth_anything_vits.pth" --probe '{bin}\libraries\video_depth_anything\video_depth_probe.exe'

Les poids restent locaux. [Référence Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything)

## Vidéos longues et flux

`stream.py` reçoit les trames RGB planaires FP32 little-endian dans l’ordre de la vidéo. Chaque trame a la forme `[3,H,W]` et utilise la normalisation ImageNet. L’appelant décode et redimensionne les trames. Le programme renvoie une carte `[H,W]` par trame réelle à la résolution du tenseur. `--input -` et `--output -` sélectionnent stdin et stdout. Les chemins désignent des fichiers binaires bruts et un fichier de sortie doit être nouveau.

Le lanceur natif conserve une session et un graphe par résolution pendant le flux. Il infère sur des fenêtres de 32 trames avec un pas de 22. L’hôte garde les dix trames d’ancrage, ajuste échelle et décalage sur les deux cartes de référence et fond huit cartes à chaque raccord. Il répète la dernière trame pour compléter une fenêtre, puis rend uniquement les cartes des trames réelles. [Découpage de référence](https://github.com/DepthAnything/Video-Depth-Anything/blob/main/video_depth_anything/video_depth.py)

```powershell
& $py -B .\libraries\video_depth_anything\stream.py --probe $probe --weights $weights --height 280 --width 420 --backend cuda --expected-device 'RTX 3090' --input .\rgb.f32le --output .\depth.f32le
```

`$weights` est le SafeTensors FP32 créé par `prepare.py --checkpoint $checkpoint --output $weights`. Ajouter `--stream --backend cuda --expected-device 'RTX 3090'` au script de validation pour comparer trois fenêtres consécutives couvrant 51 images à la chaîne PyTorch. Le script compare aussi les longueurs autour des limites de découpage.

Sous Windows x64 sur la RTX 3090 via `iree_cuda`, le rejeu a comparé les 51 cartes au calcul PyTorch avec le checkpoint Small de CyTools. L’écart absolu maximal est de `8,54e-5` sur le flux. Les essais des clips de deux et trois images donnent jusqu’à `6,92e-6`.
