# État des sources

[English](VERSION.md) · [README](README.fr.md)

## Portages de modèles

- Real-ESRGAN partage RRDBNet entre les variantes ×2, ×4 et anime, avec une ABI C et un CLI JSON.
- La validation Windows sur CPU couvre SRVGGNet, SwinIR-M JPEG10, RealESRGAN_x2plus, Video Depth Anything Small et deux checkpoints Qwen à contexte court.
- Dolphin 3.0 Llama 3.1 8B reproduit PyTorch sur CPU Windows pour deux prompts après conversion en FP16 des poids GGUF locaux.
- AudioLDM-S Full v2 génère du son mono à partir de tokens et d’un latent initial. Son encodeur CLAP texte, son ordonnanceur DDIM, son UNet, son VAE et son vocodeur HiFi-GAN ont été comparés à la référence PyTorch locale.
- La RTX 3090 a validé SwinIR JPEG10, JPEG20, ColorDN noise25 et SRVGGNet avec les checkpoints réels. AudioLDM a produit 5 152 échantillons et Video Depth Anything a traité 51 images en trois fenêtres avec leurs poids réels. [SwinIR](libraries/swinir/README.md) · [AudioLDM](libraries/audioldm/README.md) · [Video Depth Anything](libraries/video_depth_anything/README.md)
- TripoSR a produit un PLY couleur de 252 sommets et 488 triangles sur la RTX 3090 avec le checkpoint officiel. Les triplans, la densité et les couleurs ont été comparés à PyTorch. [TripoSR](libraries/triposr/README.md)
- Qwen 3.5 0.8B reproduit les tokens de Transformers sur CPU Windows. [Qwen 3.5](libraries/qwen35/README.md)
- Les encodeurs T5-XXL et UMT5-XXL concordent avec Transformers FP32 sur CPU Windows et RTX 3090 avec les checkpoints FP16 de FLUX et Wan utilisés par ComfyUI. [T5 et UMT5](libraries/t5/README.md)
- FLUX.1 Schnell utilise le checkpoint ComfyUI FP8, CLIP, T5, quatre pas d’Euler et le VAE officiel. Le débruiteur et le VAE ont été comparés à PyTorch. Sur la RTX 3090 sous Windows, le lanceur direct et la CLI ImageStudio ont produit des PNG identiques en 512 × 512 et en 1024 × 1024, ainsi qu’en 96 × 64 avec le LoRA Octree. Le rendu TAEF1 via MCP a été validé en 96 × 64. [FLUX.1 Schnell](libraries/flux1/README.md)
- Whisper Large v3 Turbo reproduit le texte, les tokens et les horodatages de Transformers avec les poids officiels sur CPU Windows et RTX 3090. AudioStudio et PersonaRP utilisent ce port pour leur opération de transcription. [Whisper](libraries/whisper/README.md#transcrire-un-fichier-audio)
- Stable Diffusion 1.5 utilise CLIP-L, le U-Net, DDIM déterministe et le décodeur VAE d’origine en FP32. Sous Windows sur la RTX 3090, la chaîne a été comparée à PyTorch et le lanceur direct et la CLI ImageStudio ont généré des PNG 512 × 512 identiques octet pour octet. [Stable Diffusion 1.5](libraries/stable_diffusion/README.md)
- SD 1.5 Inpainting ajoute l’encodeur VAE et le U-Net à neuf canaux. Sous Windows sur la RTX 3090, la sortie 192 × 128 concorde avec PyTorch à un niveau RGB8 par canal près aux forces 0,75 et 1, et conserve exactement les pixels hors masque. En 512 × 512, le lanceur ZML et la CLI ImageStudio produisent des PNG identiques octet pour octet. [SD 1.5 Inpainting](libraries/stable_diffusion/README.md#sd-15-inpainting)
- Le décodeur de structure TRELLIS donne les mêmes coordonnées occupées que PyTorch. Les flots de structure et SLAT concordent à `4,77e-7` près après quatre pas d’Euler sur CPU Windows avec des poids synthétiques réduits. [TRELLIS](libraries/trellis/README.md)
- Sur CPU Windows avec des poids synthétiques, le conditionnement DINOv2 concorde avec PyTorch à `7,16e-7` près et les trois têtes TRELLIS à `2,39e-7` près. Le test relit les trois formats 3D. [DINOv2](libraries/dinov2/README.md) · [TRELLIS](libraries/trellis/README.md)
- La chaîne TRELLIS v1 complète a été vérifiée sous Windows sur la RTX 3090 via `iree_cuda` avec des poids synthétiques. Les deux flots ont exécuté 25 pas d’Euler chacun. Les 357 voxels occupés et les faces du maillage concordent avec PyTorch, avec un écart maximal de `3,55e-6` sur les tenseurs. [Rejeu de la chaîne](libraries/trellis/README.md#chaîne-complète-avec-poids-synthétiques)
- GVHMR reconstruit les maillages SMPL-X sur 12 et 312 images canoniques de la vidéo de tennis avec les poids CyTools sous Windows. Le réseau temporel ZML tourne sur la RTX 3090 via `iree_cuda`. L’écart maximal des sommets monde face à la référence PyTorch est de `2,41e-6`. [GVHMR](libraries/gvhmr/README.md)
- GVHMR reconstruit les maillages sur 195 images à caméra mobile avec les poids locaux CyTools, de SIFT à SMPL-X sur la RTX 3090 via `iree_cuda`. À référence OpenCV/PyCOLMAP fixée, le raffinement des homographies ramène l’écart maximal de trajectoire de `6,14°` à `2,90°`. Sur le même signal caméra, l’écart maximal des sommets monde face à PyTorch est de `1,43e-5`. [Validation caméra mobile](libraries/gvhmr/README.md#raffinement-des-homographies-simplevo)
- Le rejeu GVHMR sur 195 images active aussi YOLOv8x, ViTPose-H et les jetons HMR2 en CUDA avec les poids CyTools. Les trois modèles concordent avec leurs références FP32. L’écart maximal des sommets monde est de `9,54e-6` face à PyTorch avec le calcul d’angle stabilisé et de `5,36e-3` face au pipeline amont. [Validation du prétraitement](libraries/gvhmr/README.md#prétraitement-vidéo-par-zml)
- Le décodeur UDP de ViTPose-H retourne les coordonnées et les scores dans le même graphe ZML. Sur le rejeu CUDA des 195 images, les 3 315 scores sont identiques à OpenCV sur les mêmes cartes et l’écart maximal des coordonnées est de `0,000122 px`. [Rejeu avec décodage UDP](libraries/gvhmr/README.md#prétraitement-vidéo-par-zml)
- ViTPose-H et HMR2 partagent le recadrage ZML et la normalisation ImageNet. Le rejeu à caméra mobile en CUDA jusqu’à SMPL-X a comparé les 195 recadrages FP32 à OpenCV/PyTorch avec un écart maximal de `0`. [Validation du recadrage](libraries/gvhmr/README.md#prétraitement-vidéo-par-zml)
- Le graphe commun à ViTPose-H et HMR2 met aussi les trames RGB à demi-résolution. Le rejeu CUDA des 195 images jusqu’à SMPL-X a retrouvé exactement les 134 784 000 valeurs RGB8 de PyAV. [Validation de la mise à demi-résolution](libraries/gvhmr/README.md#prétraitement-vidéo-par-zml)
- SimpleVO utilise le même filtre ZML pour ses images clés. Le test CUDA indépendant sur 195 images a retrouvé exactement les pixels de PyAV, les descripteurs SIFT et les 34 100 correspondances, puis a vérifié la trajectoire SimpleVO face à SciPy. [Validation SimpleVO](libraries/gvhmr/README.md#extraction-sift-et-appariement-en-cuda)

## Limites connues

- Le build CUDA du lanceur Qwen 3.5 dépend du module FlashInfer CUTLASS MoE du fork Windows.
- DINOv2, les flots de structure et SLAT et les trois têtes de TRELLIS v1 passent séparément avec les poids officiels sur la RTX 3090. Le rejeu de bout en bout des poids officiels reste à réaliser.
- Les chaînes complètes de génération d’images et de vidéos figurent dans l’inventaire. La validation du moteur ZML Linux dans PersonaRP Colab reste à faire.
- Les validations Windows sont détaillées par modèle et backend. La matrice de l’ABI de base comprend des essais GPU.

[Inventaire des modèles](docs/model-adapters.json) · [Validation de l’ABI Windows](tests/windows/README.md) · [Résultats de restauration](tests/windows/restoration.md) · [Résultats de génération](tests/windows/generation-parts.md)
