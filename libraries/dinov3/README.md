# DINOv3 ViT-L/16

Cet encodeur ZML produit le conditionnement utilisé par LocalMesh et TRELLIS.2.
Il attend une image normalisée ImageNet, calcule 24 blocs ViT avec RoPE sur
les patches et renvoie le jeton de classe, les quatre registres puis les
jetons image. La projection finale est une LayerNorm sans paramètres, comme
dans l’extracteur LocalMesh. Les dimensions sont des multiples de 16 de 16
à 1 024. Le chargement utilise directement le checkpoint FP32 au format
Transformers et vérifie ses 415 noms et dimensions.

Built with DINOv3. Meta contrôle l’accès aux poids. Obtenir les fichiers
locaux depuis l’éditeur du modèle. [Poids officiels](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) · [Licence DINOv3](https://ai.meta.com/resources/models-and-libraries/dinov3-license/)

Compiler la cible //libraries/dinov3:dinov3_condition avec le moteur épinglé.
Le probe accepte plusieurs couples d’entrée et de sortie par session pour
une même résolution. Les entrées sont [1,3,H,W] et les sorties
[1,5+H×W/256,1024], en FP32 little-endian contigu.

```bash
dinov3_condition schema
dinov3_condition cuda model.safetensors 512 512 image.f32 tokens.f32
python -B libraries/dinov3/validate.py \
  --probe "$probe" --model "$model_dir" --scratch "$scratch" --image "$photo" \
  --size 512 1024 --backend cuda --expected-device cuda:0 --expected-pjrt cuda \
  --reference-device cuda:0 --report "$report"
```

Le script vérifie des images rectangulaires et carrées et rejoue chaque image
dans le même modèle chargé. Il compare à Transformers en FP32 et vérifie le
périphérique de sortie. Sous Windows CUDA, utiliser le PJRT iree_cuda et le
nom du périphérique PJRT. [Configuration des DLL](../botsort/README.md#validation-cuda-sous-windows-x64)

## Validation avec le checkpoint officiel

Colab L4, CUDA PJRT et Transformers 5.22.1, calcul FP32. Les jetons
ont été comparés à PyTorch en rejouant chaque entrée dans la même session.

| Taille | Jetons de sortie | RMSE |
| --- | ---: | ---: |
| 32 × 48 | 11 | 2,068 × 10⁻⁶ |
| 512 × 512 | 1 029 | 7,789 × 10⁻⁷ |
| 1 024 × 1 024 | 4 101 | 1,571 × 10⁻⁶ |

Checkpoint facebook/dinov3-vitl16-pretrain-lvd1689m, révision
ea8dc2863c51be0a264bab82070e3e8836b02d51.
