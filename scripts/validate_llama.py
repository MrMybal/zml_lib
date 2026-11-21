"""Exercise the Llama ZML CLI with a deterministic local Transformers fixture."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM

from build import ROOT, TEMP_ROOT, run_environment, scratch_directory


VOCAB = {
    "<unk>": 0,
    "<|begin_of_text|>": 1,
    "<|end_of_text|>": 2,
    "<|start_header_id|>": 3,
    "<|end_header_id|>": 4,
    "<|eot_id|>": 5,
    "Ċ": 6,
    "user": 7,
    "assistant": 8,
    "bonjour": 9,
    "salut": 10,
}


def write_tokenizer(directory):
    tokenizer = Tokenizer(models.BPE(VOCAB, merges=[], unk_token="<unk>", ignore_merges=True))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens([token for token in VOCAB if token.startswith("<|")])
    assert tokenizer.encode("user salut assistant").ids == [7, 10, 8]
    tokenizer.save(str(directory / "tokenizer.json"))


def fixture(directory):
    write_tokenizer(directory)
    config = LlamaConfig(
        vocab_size=len(VOCAB), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
        bos_token_id=1, eos_token_id=2, tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.fill_(1 if "norm" in name else 0)
        model.model.embed_tokens.weight[:, 0] = 1
        model.lm_head.weight[VOCAB["bonjour"], 0] = 1
        logits = model(torch.tensor([[1, 3, 7, 4, 6, 10, 5, 6, 3, 8, 4, 6]])).logits
        if logits[:, -1].argmax().item() != VOCAB["bonjour"]:
            raise AssertionError("Unexpected PyTorch reference token")
    model.save_pretrained(directory, safe_serialization=True)


def cache_fixture(directory):
    write_tokenizer(directory)
    torch.manual_seed(9)
    config = LlamaConfig(
        vocab_size=len(VOCAB), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        bos_token_id=1, eos_token_id=2, tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config).eval()
    prompt = [1, 10, 7, 8, 4, 6]
    tokens = prompt.copy()
    with torch.inference_mode():
        for _ in range(3):
            logits = model(torch.tensor([tokens]), use_cache=False).logits[0, -1]
            tokens.append(logits.argmax().item())
    model.save_pretrained(directory, safe_serialization=True)
    return prompt, tokens[len(prompt):]


def run_probe(probe, directory, context, *args):
    env = run_environment(directory, ROOT.parent / "zml_win/bazel-win-tools")
    result = subprocess.run(
        [str(probe), "cpu", str(directory), str(context), *args],
        cwd=directory, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    print(result.stderr, end="", file=sys.stderr)
    result.check_returncode()
    if "backend=cpu pjrt=cpu device=cpu:0" not in result.stderr:
        raise AssertionError("Llama used an unexpected backend")
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    args = parser.parse_args()
    probe = args.probe.resolve(strict=True)
    if not probe.is_relative_to(TEMP_ROOT):
        parser.error("Run Windows validation through scripts/build.py --then")
    torch.set_num_threads(4)
    with scratch_directory() as folder:
        fixture(folder)
        result = run_probe(probe, folder, 24, "salut")
        if "bonjour" not in result:
            raise AssertionError(f"Unexpected native result {result!r}")
        print("Llama generated the expected token on cpu:0")
    with scratch_directory() as folder:
        prompt, expected = cache_fixture(folder)
        actual = json.loads(run_probe(probe, folder, len(prompt) + len(expected), "--token-ids", *map(str, prompt)))
        if actual != expected:
            raise AssertionError(f"KV-cache decode: expected {expected}, got {actual}")
        print(f"Llama KV-cache decode matched PyTorch on cpu:0: {actual}", flush=True)


if __name__ == "__main__":
    main()
