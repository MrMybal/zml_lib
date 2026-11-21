# ParseNet

Le port ZML reproduit l’encodeur résiduel, les dix blocs du corps et le décodeur du profil FaceXLib à 512 × 512, avec réflexion aux bords, interpolation au plus proche et BatchNorm d’inférence. Il prend un visage aligné en RGB planaire FP32 dans `[-1, 1]` et produit les 19 plans de logits puis les trois plans RGB auxiliaires.

Le fichier ComfyUI `facedetection/parsing_parsenet.pth` fournit 202 tenseurs FP32 d’inférence. Le convertisseur écarte les 36 compteurs BatchNorm et conserve les statistiques.

Dans le pipeline CodeFormer, les classes des logits définissent le masque sémantique utilisé pour le recollement. La validation compare les logits et la sortie RGB sur le même visage restauré par la référence. [Commande de conversion et validation](../codeformer/README.md) · [Référence FaceXLib](https://github.com/xinntao/facexlib)
