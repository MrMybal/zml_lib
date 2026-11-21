"""Inspect a local Llama GGUF or prepare a dense SafeTensors repository for ZML."""

import argparse
import json
from pathlib import Path
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build import scratch_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="omit to inspect the model and tokenizer without dequantizing")
    args = parser.parse_args()
    source = args.checkpoint.resolve(strict=True)
    if not source.is_file() or source.suffix.lower() != ".gguf":
        parser.error("Expected a local .gguf file")
    options = {"gguf_file": source.name, "local_files_only": True, "trust_remote_code": False}
    config = AutoConfig.from_pretrained(source.parent, **options)
    if config.model_type != "llama":
        parser.error(f"Expected Llama, found {config.model_type}")
    tokenizer = AutoTokenizer.from_pretrained(source.parent, **options)
    print(json.dumps({"model_type": config.model_type, "layers": config.num_hidden_layers,
                      "width": config.hidden_size, "tokens": len(tokenizer)}), flush=True)
    if not args.output:
        return
    destination = args.output.resolve()
    if destination.exists() or not destination.parent.is_dir():
        parser.error("Choose a new output directory with an existing parent")
    with scratch_directory() as work:
        if work.drive != destination.drive:
            parser.error("Choose an output directory on the same volume as _Temp for atomic publication")
        model = AutoModelForCausalLM.from_pretrained(source.parent, torch_dtype=torch.float16, **options)
        staged = work / "prepared"
        model.save_pretrained(staged, safe_serialization=True, max_shard_size="2GB")
        tokenizer.save_pretrained(staged)
        if not staged.resolve(strict=True).is_relative_to(work) or destination.exists():
            raise ValueError("Unsafe output path")
        staged.rename(destination)
    print(destination)


if __name__ == "__main__":
    main()
