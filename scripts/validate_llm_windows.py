"""Compare the Windows Llama launcher to PyTorch using a local Dolphin/Llama GGUF."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from build import ROOT, TEMP_ROOT, run_environment, scratch_directory


def reference(model_path, output):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False, dtype=torch.float16, attn_implementation="eager",
    ).eval()
    eos = model.config.eos_token_id
    stops = set(eos if isinstance(eos, list) else [eos])
    prompts = ("The capital of France is", "Continue the sequence: 1, 2, 3, 4,")
    cases = []
    with torch.inference_mode():
        for text in prompts:
            prompt = tokenizer.encode(text, add_special_tokens=False)
            if model.config.bos_token_id is not None:
                prompt.insert(0, model.config.bos_token_id)
            tokens = prompt.copy()
            for _ in range(3):
                logits = model(torch.tensor([tokens]), use_cache=False).logits[0, -1]
                token = logits.argmax().item()
                if token in stops:
                    break
                tokens.append(token)
            generated = tokens[len(prompt):]
            if not generated:
                raise ValueError("PyTorch produced no token for the validation prompt")
            cases.append({"prompt": prompt, "generated": generated, "text": tokenizer.decode(generated)})
    output.write_text(json.dumps(cases), encoding="utf-8")


def compare(probe, model_path, env, work):
    reference_path = work / "reference.json"
    subprocess.run(
        [sys.executable, "-B", str(Path(__file__).resolve()), "--reference", str(model_path), str(reference_path)],
        cwd=work, env=env, check=True,
    )
    for case in json.loads(reference_path.read_text(encoding="utf-8")):
        prompt, expected = case["prompt"], case["generated"]
        result = subprocess.run(
            [str(probe), "cpu", str(model_path), str(len(prompt) + 3), "--token-ids", *map(str, prompt)],
            cwd=work, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        print(result.stderr, end="", file=sys.stderr)
        result.check_returncode()
        if "backend=cpu pjrt=cpu device=cpu:0" not in result.stderr:
            raise ValueError("Llama used an unexpected backend")
        actual = json.loads(result.stdout)
        if actual != expected:
            raise AssertionError(f"Expected {expected} for {prompt}, got {actual}")
        print(f"Dolphin/Llama cpu:0 {expected} -> {ascii(case['text'])}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-gguf", type=Path, help="GGUF Llama checkpoint already on this computer")
    parser.add_argument("--reference", type=Path, nargs=2, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.reference:
        reference(*args.reference)
        return
    if not args.llama_gguf or not os.environ.get("ZML_BIN_DIR"):
        parser.error("Run through scripts/build.py --then and supply --llama-gguf")
    probe = (Path(os.environ["ZML_BIN_DIR"]) / "libraries/llama/llama_generate.exe").resolve(strict=True)
    if not probe.is_relative_to(TEMP_ROOT):
        parser.error("The launcher must be built in the configured temporary directory")
    source = args.llama_gguf.resolve(strict=True)
    tools = ROOT.parent / "zml_win/bazel-win-tools"
    subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts/validate_llama.py"), "--probe", str(probe)],
        cwd=ROOT, check=True,
    )
    with scratch_directory() as work:
        env = run_environment(work, tools)
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        model_path = work / "llama"
        subprocess.run(
            [sys.executable, "-B", str(ROOT / "libraries/llama/prepare_gguf.py"),
             "--checkpoint", str(source), "--output", str(model_path)],
            cwd=work, env=env, check=True,
        )
        compare(probe, model_path, env, work)


if __name__ == "__main__":
    main()
