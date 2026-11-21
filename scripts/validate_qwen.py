"""Compare greedy ZML and PyTorch tokens for a local Qwen checkpoint."""

import argparse
from pathlib import Path
import subprocess
import sys

from build import ROOT, TEMP_ROOT, scratch_directory


def reference(directory, profile, prompts, steps):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.set_num_threads(4)
    config = AutoConfig.from_pretrained(directory, local_files_only=True)
    if profile == "qwen3":
        # The local snapshot supplies an explicit lm_head.weight in the second shard.
        config.tie_word_embeddings = False
    model = AutoModelForCausalLM.from_pretrained(
        directory, config=config, local_files_only=True, dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    eos = {151645, 151643} if profile == "qwen3" else {151645}
    results = []
    with torch.inference_mode():
        for prompt_ids in prompts:
            generated = []
            ids = torch.tensor([prompt_ids])
            for _ in range(steps):
                token = model(ids, use_cache=False).logits[0, -1].argmax().item()
                generated.append(token)
                if token in eos:
                    break
                ids = torch.cat((ids, torch.tensor([[token]])), dim=1)
            results.append(generated)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("qwen2", "qwen3"), required=True)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=3, choices=range(1, 9))
    args = parser.parse_args()
    model = args.model.resolve(strict=True)
    probe = args.probe.resolve(strict=True)
    if not probe.is_relative_to(TEMP_ROOT):
        parser.error("Run Windows validation through scripts/build.py --then")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    sys.path.insert(0, str(ROOT))
    from libraries.qwen_decoder.prepare import export, inspect_model

    layout, sources, _ = inspect_model(model, args.profile)
    with scratch_directory() as work:
        prompts = ("Bonjour, je suis", "The capital of France is")
        tokenized = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
        references = reference(model, args.profile, tokenized, args.steps)
        weights = work / "weights.safetensors"
        export(layout, sources, weights)
        for prompt, prompt_ids, expected in zip(prompts, tokenized, references, strict=True):
            result = subprocess.run(
                [str(probe), "cpu", str(weights), "32", str(args.steps), *map(str, prompt_ids)],
                cwd=ROOT, capture_output=True, text=True, errors="replace",
            )
            print(result.stderr, end="", file=sys.stderr)
            result.check_returncode()
            actual = [int(line) for line in result.stdout.splitlines()]
            placed = result.stderr.count("backend=cpu pjrt=cpu device=cpu:0 shards=1")
            if actual != expected or placed != len(actual):
                raise AssertionError(f"{prompt!r}: expected {expected}, got {actual} ({placed} placements)")
            print(f"{args.profile} cpu:0 {prompt!r}: {actual}", flush=True)


if __name__ == "__main__":
    main()
