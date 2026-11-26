"""Compare repeated ZML TDT steps to PyTorch LSTM and joint scores."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
from safetensors.torch import load_file
import torch
from torch.nn import functional as F

from prepare import TDT, inspect

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.probe_placement import verify_placement


def step(weights, lstm, token, frame, state):
    value = weights["decoder.embedding.weight"][token].reshape(1, 1, 640)
    h, c = state[0::2].unsqueeze(1).contiguous(), state[1::2].unsqueeze(1).contiguous()
    predicted, (hidden, cell) = lstm(value, (h, c))
    encoded = F.linear(frame, weights["encoder_projector.weight"], weights["encoder_projector.bias"])
    decoded = F.linear(predicted, weights["decoder.decoder_projector.weight"], weights["decoder.decoder_projector.bias"])
    logits = F.linear((encoded + decoded).relu(), weights["joint.head.weight"], weights["joint.head.bias"])
    next_state = torch.stack((hidden, cell), dim=1).flatten()
    return torch.cat((logits.flatten(), next_state))


def validate(args, work):
    checkpoint = work / "decoder.safetensors"
    source = args.model / "model.fermion" if args.model.is_dir() else args.model
    print(json.dumps(inspect(source, "tdt", checkpoint)), flush=True)
    names = subprocess.check_output([args.probe, "schema"], text=True)
    schema = {name: tuple(map(int, dims)) for name, *dims in (line.split("\t") for line in names.splitlines())}
    if schema != TDT:
        raise ValueError("Phonon native and converter tensor schemas differ")
    weights = load_file(checkpoint, device=args.reference_device)
    lstm = torch.nn.LSTM(640, 640, num_layers=2, batch_first=True, device=args.reference_device).eval()
    prefix = "decoder.lstm."
    lstm.load_state_dict({name.removeprefix(prefix): value for name, value in weights.items() if name.startswith(prefix)})
    gen = torch.Generator().manual_seed(2369)
    state = torch.zeros((4, 640), device=args.reference_device)
    pairs, expected = [], []
    for index, token in enumerate((8192, 1166, 1987)):
        frame = torch.randn(1024, generator=gen) * 0.2
        packed = torch.cat((torch.tensor([token], dtype=torch.float32), frame, state.cpu().flatten()))
        in_file, out_file = work / f"in-{index}.f32", work / f"out-{index}.f32"
        packed.numpy().astype("<f4", copy=False).tofile(in_file)
        pairs.extend((in_file, out_file))
        value = step(weights, lstm, token, frame.to(args.reference_device), state)
        expected.append(value.cpu().numpy())
        state = value[8198:].reshape(4, 640)
    del weights, lstm
    if args.reference_device.startswith("cuda"):
        torch.cuda.empty_cache()
    proc = subprocess.run([args.probe, args.backend, checkpoint, *pairs], capture_output=True, text=True, encoding="utf-8")
    if proc.returncode:
        raise RuntimeError(proc.stderr[-6000:])
    placement = verify_placement(proc.stderr, args.backend, len(expected), args.expected_device, expected_pjrt=args.expected_pjrt)
    for sample, (out_file, reference) in enumerate(zip(pairs[1::2], expected, strict=True)):
        actual = np.fromfile(out_file, dtype="<f4")
        if actual.shape != reference.shape or not np.isfinite(actual).all():
            raise ValueError(f"Unexpected output in step {sample}")
        for name, slc in (("token", slice(8193)), ("duration", slice(8193, 8198)), ("states", slice(8198, None))):
            observed, target = actual[slc], reference[slc]
            rmse = float(np.square(observed.astype(np.float64) - target).mean()**0.5)
            print(json.dumps({"step": sample, "output": name, "rmse": rmse, "placement": placement}), flush=True)
            if rmse > 0.001 or (name != "states" and observed.argmax() != target.argmax()):
                raise ValueError(f"Phonon {name} does not match PyTorch in step {sample}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--expected-device", required=True)
    parser.add_argument("--expected-pjrt")
    parser.add_argument("--reference-device", default="cpu")
    args = parser.parse_args()
    args.model = args.model.resolve(strict=True)
    args.probe = args.probe.resolve(strict=True)
    args.scratch = args.scratch.resolve(strict=True)
    if args.scratch.drive.upper() == "C:":
        parser.error("--scratch must be outside C:")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with torch.inference_mode(), TemporaryDirectory(prefix="phonon-tdt-", dir=args.scratch) as temp:
        validate(args, Path(temp))


if __name__ == "__main__":
    main()
