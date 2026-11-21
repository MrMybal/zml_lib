# SRVGGNet Compact

Le port implémente le profil CyTools `realesr-general-x4v3`. Il valide 101 tenseurs, puis enchaîne 33 convolutions suivies de PReLU, une convolution de sortie et un PixelShuffle ×4. Le réseau ajoute le résidu à l’image agrandie au plus proche voisin. [Architecture Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/archs/srvgg_arch.py)

`Model.infer` reçoit une image RGB planaire FP32 `[1,3,H,W]` dans `[0,1]` et produit les valeurs brutes `[1,3,4H,4W]`. Le modèle réutilise ses poids et son exécutable pour des dimensions identiques. L’appelant détient les buffers et libère le modèle avant la session.

La référence PyTorch a chargé `CyTools_VideoStudio/runtime/restoration-validation/realesr-general-x4v3.pth`, SHA-256 `8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292`. Sur Windows x64 et `cpu:0`, l’écart absolu maximal est de `3,22e-6` sur deux tailles. Deux inférences consécutives ont passé le contrôle de placement pour chaque taille. [Rejeu Windows](../../tests/windows/restoration.md)
