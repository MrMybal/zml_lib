<table width="100%">
  <tr>
    <td width="300" align="left" valign="top">
      <a href="https://github.com/MrMybal/zml_lib">Dépôt</a><br>
      <a href="docs/model-adapters.json">Inventaire des modèles</a><br><br>
      <a href="README.md">English</a> · <a href="README.fr.md">Français</a>
    </td>
    <td width="400" align="center" valign="middle"><strong>Zig · ZML</strong></td>
    <td width="300" align="right" valign="top">
      <a href="LICENSE">Licence MIT</a><br>
      <a href="VERSION.fr.md">État des sources</a><br>
      <strong>Windows x64</strong>
    </td>
  </tr>
</table>

<h1 align="center">zml_lib</h1>

Exécutez des modèles d’IA en local en Zig avec ZML. Chaque bibliothèque fournit son graphe, une cible Bazel et les outils de conversion ou de validation de ses checkpoints.

> Les validations Windows x64 couvrent notamment la restauration, AudioLDM, TripoSR et Video Depth Anything sur la RTX 3090, ainsi que Qwen 3.5 et les graphes Whisper sur CPU. [État des sources et limites connues](VERSION.fr.md)

## Bibliothèques de modèles

| Usage | Bibliothèques |
| --- | --- |
| Restauration et agrandissement d’image | [Real-ESRGAN](libraries/realesrgan/README.md), [SRVGGNet](libraries/srvggnet/README.md), [SwinIR-M JPEG10, JPEG20 et ColorDN](libraries/swinir/README.md), [RealPLKSR](libraries/realplksr/README.md) |
| Restauration faciale | [RetinaFace](libraries/retinaface/README.md), [ParseNet](libraries/parsenet/README.md), [CodeFormer](libraries/codeformer/README.md) |
| Détection et segmentation | [YOLOv8](libraries/yolov8/README.md) pour les visages, mains et personnes, [SAM](libraries/sam/README.md), [SAM 2](libraries/sam2/README.md) avec suivi d’un objet |
| Profondeur et caractéristiques visuelles | [DINOv2](libraries/dinov2/README.md), [Depth Anything V2 Large](libraries/depth_anything_v2/README.md), [Video Depth Anything Small](libraries/video_depth_anything/README.md) |
| Texte et composants image | [Qwen 2.5 0.5B](libraries/qwen2/README.md), [Qwen 3 1.7B](libraries/qwen3/README.md), [Conditionnement Qwen2.5-VL](libraries/qwen25_vl/README.md), [CLIP texte](libraries/clip/README.md), [Encodeurs T5 et UMT5](libraries/t5/README.md), [TAEF1](libraries/taesd/README.md) |
| Génération et édition d’images | [FLUX.1 Schnell et Canny Dev](libraries/flux1/README.md), [Stable Diffusion 1.5](libraries/stable_diffusion/README.md), [Qwen Image Edit 2511](libraries/qwen_image/README.md) |
| Génération audio | [AudioLDM](libraries/audioldm/README.md), [CLAP texte](libraries/laion_clap/README.md), [HiFi-GAN](libraries/hifigan/README.md) |
| Synthèse vocale | [Qwen3-TTS 12 Hz](libraries/qwen3_tts/README.md), voix préréglées, création de voix et clonage |
| Reconnaissance vocale | [Whisper, transcription avec segments horodatés](libraries/whisper/README.md), [Phonon 2 et 2.5](libraries/phonon/README.md) |
| Reconstruction 3D et mouvement | [TripoSR](libraries/triposr/README.md) image vers maillage, [TRELLIS v1](libraries/trellis/README.md) conditionnement DINOv2, structure, SLAT et décodeurs 3D, [GVHMR mouvement corporel depuis une vidéo](libraries/gvhmr/README.md), [ViTPose-H](libraries/vitpose/README.md), [caractéristiques d’apparence HMR2](libraries/hmr2/README.md), [SMPL-X maillage et articulations](libraries/smplx/README.md) |

L’inventaire recense les checkpoints, les dépendances et les portages restants. Sur CPU Windows, Llama reproduit PyTorch avec le GGUF Dolphin 3 converti en FP16, et Qwen 3.5 reproduit les tokens de Transformers avec le checkpoint officiel 0.8B. [Inventaire](docs/model-adapters.json) · [Llama](libraries/llama/README.md) · [Qwen 3.5](libraries/qwen35/README.md)

**Conditionnement d’image et reconstruction 3D**

[BiRefNet-HR](libraries/birefnet/README.md), [DINOv3](libraries/dinov3/README.md), [Depth Anything 3](libraries/depth_anything_v3/README.md), [MoGe-2](libraries/moge2/README.md), [NAF](libraries/naf/README.md), [TRELLIS.2](libraries/trellis2/README.md), [TRELLIS.2 VAE](libraries/trellis2_vae/README.md).

## Intégration

Le [catalogue local](docs/model-catalog.md) inventorie les dossiers de modèles ComfyUI et prépare les checkpoints Real-ESRGAN et Stable Diffusion 1.5, les encodeurs CLIP-L et T5-XXL ainsi que le décodeur VAE de FLUX.1.

Utilisez [l’en-tête C](include/zml_lib.h) et [l’exemple C++](examples/cpp/main.cpp) pour charger un modèle par session `zml_lib.dll` et traiter plusieurs images. L’appelant fournit les buffers hôtes et sérialise les appels à une session. Le moteur libère les sorties du périphérique après leur copie vers l’hôte. L’[ABI de restauration faciale](libraries/codeformer/zml_faces.h) peut charger RetinaFace, ParseNet et CodeFormer dans la même session.

CyTools ImageStudio peut appeler le CLI Real-ESRGAN avec une requête JSON UTF-8 sur stdin. Il reçoit une réponse JSON sur stdout et les diagnostics sur stderr. [Formats des requêtes et des buffers](libraries/realesrgan/README.md)

---

## Installation

Compilez depuis les sources sur Windows x64 avec PowerShell 7, Python 3.11, Git Bash, Bazel **9.3.0**, Zig **0.17.0** et un checkout local de `zml_win`. Préparez les plugins PJRT selon le guide Windows du fork. Fournissez les checkpoints depuis votre bibliothèque locale et suivez les instructions de conversion de chaque portage. [Versions de la chaîne de compilation](zml.lock.json) · [Guide Windows](https://github.com/MrMybal/zml_win/blob/windows-x64/docs/windows-x64.md)

Lancez cette commande depuis la racine du dépôt, avec `zml_win` dans un dossier voisin.

```powershell
python .\scripts\build.py --engine ..\zml_win -- --jobs=12
```

Le script de compilation place par défaut les fichiers de travail de Bazel et Zig sous `_Temp`. Définissez `ZML_TEMP_ROOT` sur un dossier absolu pour choisir un autre emplacement. Le dossier temporaire doit être sur un lecteur autre que `C:`. Le script supprime le dossier de l’exécution à la fin de la commande. Une compilation seule copie les binaires dans `validation-output\bin`. Avec `--then`, le script lance le test sur les binaires temporaires. La commande du test reçoit `ZML_BIN_DIR` et `ZML_TEST_DIR`. Le marqueur `{bin}` désigne le dossier des binaires.

Le moteur accepte `cpu`, `cuda`, `rocm`, `vulkan` ou `auto`. Définissez `COMFYUI_ROOT` et `CYTOOLS_ROOT` sur les dossiers d’installation locaux avant de lancer les exemples de validation qui les utilisent. [Tests de l’ABI Windows](tests/windows/README.md) · [Tests de restauration](tests/windows/restoration.md) · [Tests de génération](tests/windows/generation-parts.md)

---

## Licence

Le code original de zml_lib est sous [licence MIT](LICENSE). Les composants tiers et les poids des modèles conservent leurs conditions respectives.

| Composant | Licence ou conditions | Périmètre |
| --- | --- | --- |
| ZML et `zml_win` | [Apache-2.0](https://github.com/MrMybal/zml_win/blob/windows-x64/LICENSE) | Moteur d’inférence externe |
| BasicSR | [Apache-2.0](libraries/realesrgan/third_party/LICENSE-BasicSR.txt) | Référence RRDBNet |
| Real-ESRGAN | [BSD-3-Clause](libraries/realesrgan/third_party/LICENSE-Real-ESRGAN.txt) | Référence du modèle et code amont |
| stb | [MIT ou alternative du domaine public](libraries/realesrgan/third_party/LICENSE-stb.txt) | Codecs d’image inclus |
| FlexiCubes | [Apache-2.0](libraries/trellis/flexicubes/LICENSE.txt) | Tables de topologie de surface pour TRELLIS |
| Poids des modèles | [Provenance et conditions amont](libraries/realesrgan/models/provenance.json) | Checkpoints locaux. Voir les sources indiquées par chaque bibliothèque. |

## Références

[ZML](https://github.com/zml/zml) · [Fork Windows](https://github.com/MrMybal/zml_win) · [Inventaire et ordre des portages](docs/model-adapters.json)
