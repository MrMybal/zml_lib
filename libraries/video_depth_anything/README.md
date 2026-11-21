# Video Depth Anything Small

Le modèle réutilise [DINOv2 ViT-S/14](../dinov2/README.md) et la tête DPT. Quatre modules d’attention temporelle relient jusqu’à 32 images d’un clip. `Model.infer` reçoit RGB FP32 normalisé `[T,3,H,W]` et produit une profondeur relative `[T,H,W]`. `H` et `W` sont des multiples de 14. Le découpage des vidéos longues et le mode streaming restent à intégrer.

Le checkpoint local de CyTools VideoToStereo3D contient 351 tenseurs FP32. Sous Windows x64 sur `cpu:0`, l’erreur absolue maximale contre PyTorch est de `5.96046448e-6` sur un clip de deux images 28 × 42 et un clip de trois images 42 × 28.

    $stereo = (Join-Path $env:CYTOOLS_ROOT 'CyTools_VideoToStereo3D')
    $py = Join-Path $stereo 'runtime\engine\Scripts\python.exe'
    $env:ZML_CPU_PJRT_DLL = (Resolve-Path '..\zml_win\bazel-win-tools\pjrt-cpu\pjrt_cpu.dll').Path
    python .\scripts\build.py --target //libraries/video_depth_anything:video_depth_probe -- --jobs=8 --then $py -B .\libraries\video_depth_anything\validate.py --reference "$stereo\runtime\source\video-depth-anything" --checkpoint "$stereo\data\models\depth-vda-small\video_depth_anything_vits.pth" --probe '{bin}\libraries\video_depth_anything\video_depth_probe.exe'

Les poids restent locaux. [Référence Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything)
