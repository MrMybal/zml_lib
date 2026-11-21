<table width="100%">
  <tr>
    <td width="300" align="left" valign="top">
      <a href="https://github.com/MrMybal/zml_lib">Repository</a><br>
      <a href="docs/model-adapters.json">Model inventory</a><br><br>
      <a href="README.md">English</a> · <a href="README.fr.md">Français</a>
    </td>
    <td width="400" align="center" valign="middle"><strong>Zig · ZML</strong></td>
    <td width="300" align="right" valign="top">
      <a href="LICENSE">MIT License</a><br>
      <a href="VERSION.md">Source status</a><br>
      <strong>Windows x64</strong>
    </td>
  </tr>
</table>

<h1 align="center">zml_lib</h1>

Run local AI models in Zig with ZML. Each library contains its model graph, a Bazel target, and the conversion or validation tools for its supported checkpoints.

> Windows x64 CPU validation covers the latest restoration ports, two short-context Qwen profiles, and AudioLDM from prompt tokens to audio. GPU coverage varies by port. [Source status and known limits](VERSION.md)

## Model libraries

| Use | Libraries |
| --- | --- |
| Image restoration and upscaling | [Real-ESRGAN](libraries/realesrgan/README.md), [SRVGGNet](libraries/srvggnet/README.md), [SwinIR-M JPEG10](libraries/swinir/README.md), [RealPLKSR](libraries/realplksr/README.md) |
| Face restoration | [RetinaFace](libraries/retinaface/README.md), [ParseNet](libraries/parsenet/README.md), [CodeFormer](libraries/codeformer/README.md) |
| Detection and segmentation | [YOLOv8](libraries/yolov8/README.md) for faces, hands and people, [SAM](libraries/sam/README.md), [SAM 2](libraries/sam2/README.md) with single-object tracking |
| Depth and visual features | [DINOv2](libraries/dinov2/README.md), [Depth Anything V2 Large](libraries/depth_anything_v2/README.md), [Video Depth Anything Small](libraries/video_depth_anything/README.md) |
| Text and image components | [Qwen 2.5 0.5B](libraries/qwen2/README.md), [Qwen 3 1.7B](libraries/qwen3/README.md), [CLIP text](libraries/clip/README.md), [TAEF1](libraries/taesd/README.md) |
| Audio generation | [AudioLDM](libraries/audioldm/README.md), [CLAP text](libraries/laion_clap/README.md), [HiFi-GAN](libraries/hifigan/README.md) |

The [model inventory](docs/model-adapters.json) records checkpoints, dependencies, and remaining ports. [Llama](libraries/llama/README.md) matched PyTorch on Windows CPU with the local Dolphin 3 GGUF converted to FP16. The [Qwen 3.5](libraries/qwen35/README.md) launcher builds on Windows and awaits a local checkpoint for inference validation.

## Integration

Use [the C header](include/zml_lib.h) and [C++ example](examples/cpp/main.cpp) to load one model per `zml_lib.dll` session and process multiple images. The caller supplies the host buffers and serializes calls to a session. The runtime releases device outputs after copying them to the host. The [face-restoration ABI](libraries/codeformer/zml_faces.h) can hold RetinaFace, ParseNet, and CodeFormer in one session.

CyTools ImageStudio can call the Real-ESRGAN CLI with one UTF-8 JSON request on stdin. It receives one JSON response on stdout and diagnostics on stderr. [Request and buffer formats](libraries/realesrgan/README.md)

---

## Installation

Build from source on Windows x64 with PowerShell 7, Python 3.11, Git Bash, Bazel **9.3.0**, Zig **0.17.0**, and a local checkout of `zml_win`. Prepare the PJRT plugins according to the fork's Windows guide. Supply checkpoints from your local model library and use the conversion instructions for each port. [Toolchain pin](zml.lock.json) · [Windows guide](https://github.com/MrMybal/zml_win/blob/windows-x64/docs/windows-x64.md)

Run this command from the repository root with `zml_win` as a sibling checkout.

```powershell
python .\scripts\build.py --engine ..\zml_win -- --jobs=12
```

The build script places its Bazel and Zig work files under the repository's `_Temp` directory by default. Set `ZML_TEMP_ROOT` to an absolute directory to use a separate location. The temporary directory must be on a drive other than `C:`. The script removes the per-run directory when the command ends. A build copies the resulting binaries to `validation-output\bin`. With `--then`, the script runs the test against the temporary binaries. Test commands receive `ZML_BIN_DIR` and `ZML_TEST_DIR`, and `{bin}` expands to the bin directory.

The runtime accepts `cpu`, `cuda`, `rocm`, `vulkan`, or `auto`. Set `COMFYUI_ROOT` and `CYTOOLS_ROOT` to the local installation directories before running the validation examples that use them. [Windows ABI tests](tests/windows/README.md) · [Restoration tests](tests/windows/restoration.md) · [Generation tests](tests/windows/generation-parts.md)

---

## License

The original zml_lib code is available under the [MIT License](LICENSE). Third-party components and model checkpoints retain their own terms.

| Component | License or terms | Scope |
| --- | --- | --- |
| ZML and `zml_win` | [Apache-2.0](https://github.com/MrMybal/zml_win/blob/windows-x64/LICENSE) | External inference engine |
| BasicSR | [Apache-2.0](libraries/realesrgan/third_party/LICENSE-BasicSR.txt) | RRDBNet reference |
| Real-ESRGAN | [BSD-3-Clause](libraries/realesrgan/third_party/LICENSE-Real-ESRGAN.txt) | Model reference and upstream code |
| stb | [MIT or public-domain alternative](libraries/realesrgan/third_party/LICENSE-stb.txt) | Included image codecs |
| Model weights | [Provenance and upstream terms](libraries/realesrgan/models/provenance.json) | Local checkpoints. See each library for its sources. |

## References

[ZML](https://github.com/zml/zml) · [Windows fork](https://github.com/MrMybal/zml_win) · [Model inventory and port order](docs/model-adapters.json)
