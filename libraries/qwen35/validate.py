"""Verify greedy text generation from a local Qwen 3.5 SafeTensors checkpoint."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build import ROOT, run_environment, scratch_directory
from probe_placement import PLATFORMS


def reference(directory):
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    model = AutoModelForImageTextToText.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16, attn_implementation="eager",
    ).eval()
    stops = model.generation_config.eos_token_id
    if not isinstance(stops, list):
        stops = [stops]
    cases = []
    with torch.inference_mode():
        for text in ("The capital of France is", "Continue the sequence: 1, 2, 3, 4,"):
            prompt = tokenizer.encode(text, add_special_tokens=False)
            tokens = prompt.copy()
            for _ in range(3):
                logits = model(input_ids=torch.tensor([tokens]), use_cache=False).logits[0, -1]
                token = logits.argmax().item()
                if token in stops:
                    break
                tokens.append(token)
            expected = tokens[len(prompt):]
            if len(expected) != 3:
                raise ValueError(f"Reference stopped early for {text!r}")
            cases.append((prompt, expected, tokenizer.decode(expected)))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend", choices=PLATFORMS, default="cpu")
    parser.add_argument("--expected-device")
    args = parser.parse_args()
    probe = args.probe.resolve(strict=True)
    model = args.model.resolve(strict=True)
    tools = ROOT.parent / "zml_win" / "bazel-win-tools"
    with scratch_directory() as work:
        env = run_environment(work, tools)
        env["CUDA_CACHE_DISABLE"] = "1"
        for prompt, expected, decoded in reference(model):
            command = [probe, args.backend, model, str(len(prompt) + 3), "--token-ids", *map(str, prompt)]
            result = subprocess.run(command, cwd=work, env=env, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace")
            print(result.stderr, end="", file=sys.stderr)
            result.check_returncode()
            match = re.search(r"backend=(\w+) pjrt=(\w+) device=(.+)$", result.stderr, re.MULTILINE)
            if not match or match.group(1, 2) != (args.backend, PLATFORMS[args.backend]):
                raise AssertionError("Unexpected Qwen 3.5 backend")
            if args.expected_device and args.expected_device.casefold() not in match.group(3).casefold():
                raise AssertionError(f"Unexpected Qwen 3.5 device {match.group(3)}")
            actual = json.loads(result.stdout)
            if actual != expected:
                raise AssertionError(f"Token divergence for {prompt}: expected {expected}, got {actual}")
            print(f"Qwen 3.5 {match.group(3)} {actual} -> {ascii(decoded)}", flush=True)


if __name__ == "__main__":
    main()
