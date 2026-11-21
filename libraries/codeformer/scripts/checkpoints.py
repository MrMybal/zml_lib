"""Convert the three local ComfyUI face-restoration checkpoints to FP32 SafeTensors."""

import argparse
import hashlib
from pathlib import Path
import tempfile


SOURCES = {
    "retinaface": ("facedetection/detection_Resnet50_Final.pth", 383),
    "parsenet": ("facedetection/parsing_parsenet.pth", 202),
    "codeformer": ("facerestore_models/codeformer.pth", 515),
}


def read_original(model, model_root):
    import torch

    relative, _ = SOURCES[model]
    path = model_root / relative
    payload = torch.load(path, map_location="cpu", weights_only=True)
    group = "params_ema" if model == "codeformer" else "state_dict"
    weights = payload[group] if model == "codeformer" else payload
    normalized = {}
    for key, value in weights.items():
        name = key.removeprefix("module.") if model == "retinaface" else key
        if name in normalized:
            raise ValueError(f"Duplicate tensor {name}")
        normalized[name] = value
    return normalized, path, group


def inference_state(weights, model):
    import torch

    state = {}
    for name, value in weights.items():
        if name.endswith(".num_batches_tracked"):
            if value.shape != () or value.dtype != torch.int64:
                raise ValueError(f"Invalid BatchNorm counter {name}")
            continue
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.layout != torch.strided:
            raise ValueError(f"Expected FP32 tensor {name}")
        state[name] = value.detach().contiguous()
    expected = SOURCES[model][1]
    if len(state) != expected:
        raise ValueError(f"{model}: expected {expected} tensors, got {len(state)}")
    return state


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    from safetensors.torch import save_file

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
        staged = Path(temp) / "weights"
        staged.mkdir()
        for model in SOURCES:
            original, path, group = read_original(model, args.models)
            state = inference_state(original, model)
            save_file(state, staged / f"{model}.safetensors", {
                "model": model, "checkpoint_key": group, "checkpoint_sha256": sha256(path), "precision": "float32",
            })
            print(f"{model}: {len(state)} FP32 tensors from {group}")
        staged.rename(args.output)


if __name__ == "__main__":
    main()
