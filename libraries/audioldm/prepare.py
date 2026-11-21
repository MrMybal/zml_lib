"""Extract the local AudioLDM VAE decoder and CLAP text weights into SafeTensors."""

import argparse
from pathlib import Path
import sys

from safetensors import safe_open
from safetensors.torch import save_file


def save_subset(source, destination, include, expected):
    with safe_open(source, framework="pt", device="cpu") as weights:
        chosen = [name for name in weights.keys() if include(name)]
        if len(chosen) != expected or any(weights.get_slice(name).get_dtype() != "F32" for name in chosen):
            raise ValueError(f"{source.name}: expected {expected} FP32 parameters in the selected subset")
        save_file({name: weights.get_tensor(name) for name in chosen}, destination)


def save_decoder(model, destination):
    save_subset(model / "vae/diffusion_pytorch_model.safetensors", destination,
                lambda name: name.startswith(("decoder.", "post_quant_conv.")), 114)


def save_text(model, destination):
    buffers = {"text_model.embeddings.position_ids", "text_model.embeddings.token_type_ids"}
    save_subset(model / "text_encoder/model.safetensors", destination, lambda name: name not in buffers, 203)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory for the two extracted checkpoints")
    args = parser.parse_args()
    model = args.model.resolve(strict=True)
    destination = args.output.resolve()
    if destination.exists() or not destination.parent.is_dir():
        parser.error("Choose a new output directory with an existing parent")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from build import scratch_directory

    with scratch_directory() as work:
        if work.drive != destination.drive:
            parser.error("The output directory must be on the _Temp drive for atomic publication")
        staged = work / "prepared"
        staged.mkdir()
        save_decoder(model, staged / "vae_decoder.safetensors")
        save_text(model, staged / "clap_text.safetensors")
        staged.rename(destination)
    print(destination)


if __name__ == "__main__":
    main()
