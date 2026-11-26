"""Inspect or expand the local Phonon 2 family model.fermion into SafeTensors."""

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from math import prod
from pathlib import Path
import struct
from tempfile import TemporaryDirectory

import numpy as np


FORMAT = "fermion-five-value-parakeet-v1"
TRITS = np.arange(243, dtype=np.uint16)[:, None] // np.array([1, 3, 9, 27, 81]) % 3
TDT = {"encoder_projector.weight": (640, 1024), "encoder_projector.bias": (640,),
       "decoder.embedding.weight": (8193, 640), "decoder.decoder_projector.weight": (640, 640),
       "decoder.decoder_projector.bias": (640,), "joint.head.weight": (8198, 640), "joint.head.bias": (8198,)}
for layer in range(2):
    for side in ("ih", "hh"):
        TDT[f"decoder.lstm.weight_{side}_l{layer}"] = (2560, 640)
        TDT[f"decoder.lstm.bias_{side}_l{layer}"] = (2560,)


@dataclass(frozen=True)
class Record:
    name: str
    kind: str
    shape: tuple
    size: int
    offset: int


def index(stream):
    prefix = stream.read(8)
    if len(prefix) != 8 or not 0 < (length := int.from_bytes(prefix, "little")) <= 1024**2:
        raise ValueError("Expected the header of an extracted model.fermion")
    header = json.loads(stream.read(length))
    if header.get("format") != FORMAT:
        raise ValueError(f"Expected a {FORMAT} checkpoint")
    end = stream.seek(0, 2)
    offset = length + 8
    records = {}
    for entry in header["index"]:
        kind = entry["k"]
        name, shape, size = entry["n"], tuple(entry["shape"]), entry["b"]
        if kind == "five_value":
            name += ".weight"
        if (not name or name in records or kind not in {"five_value", "int6", "int8", "fp16"}
                or any(type(dim) is not int or dim <= 0 for dim in shape)
                or type(size) is not int or size <= 0 or offset + size > end):
            raise ValueError(f"Invalid Phonon tensor record for {name!r}")
        records[name] = Record(name, kind, shape, size, offset)
        offset += size
    if offset != end:
        raise ValueError("Unexpected bytes after the last Phonon tensor")
    return records


def expand(stream, record):
    stream.seek(record.offset)
    blob = stream.read(record.size)
    if len(blob) != record.size:
        raise ValueError(f"Truncated tensor {record.name}")
    shape = record.shape
    size = prod(shape)
    kind = record.kind
    if kind == "fp16":
        if len(blob) != size * 2:
            raise ValueError(f"Invalid FP16 length for {record.name}")
        return np.frombuffer(blob, dtype="<f2").reshape(shape)
    if not shape:
        raise ValueError(f"Quantized tensor {record.name} has no rows")
    rows = shape[0]
    if kind == "five_value":
        if len(shape) != 2:
            raise ValueError(f"Expected a five-value matrix for {record.name}")
        cols = shape[1]
        packed_cols = (cols + 4) // 5
        trit_end = rows * packed_cols
        if len(blob) < trit_end + 4 * rows:
            raise ValueError(f"Truncated five-value matrix {record.name}")
        trits = np.frombuffer(blob, np.uint8, count=trit_end)
        if np.any(trits >= len(TRITS)):
            raise ValueError(f"Invalid trits in {record.name}")
        sign = TRITS[trits].reshape(rows, packed_cols * 5)[:, :cols].astype(np.int8) - 1
        nonzero = sign != 0
        flags = int(nonzero.sum())
        scale_begin = trit_end + (flags + 7) // 8
        if len(blob) != scale_begin + 4 * rows:
            raise ValueError(f"Invalid five-value flag count for {record.name}")
        high = np.zeros(shape, dtype=bool)
        high[nonzero] = np.unpackbits(np.frombuffer(blob[trit_end:scale_begin], np.uint8), count=flags, bitorder="little")
        scales = np.frombuffer(blob, dtype="<f2", offset=scale_begin).reshape(2, rows)
        return sign * np.where(high, scales[1, :, None], scales[0, :, None])
    body_size = size if kind == "int8" else 3 * ((size + 3) // 4)
    if len(blob) != body_size + rows * 2:
        raise ValueError(f"Invalid {kind} length for {record.name}")
    if kind == "int8":
        quantized = np.frombuffer(blob, np.int8, count=size)
    else:
        chunks = np.frombuffer(blob, np.uint8, count=body_size).reshape(-1, 3).astype(np.uint32)
        words = chunks @ np.array([1, 256, 65536], dtype=np.uint32)
        codes = (words[:, None] >> np.arange(0, 24, 6, dtype=np.uint32)) & 63
        quantized = codes.reshape(-1)[:size].astype(np.int8) - 32
    scales = np.frombuffer(blob, dtype="<f2", offset=body_size).astype(np.float32)
    return (quantized.reshape(rows, -1) * scales[:, None]).reshape(shape)


def decoder_records(records):
    result = {}
    for name, shape in TDT.items():
        if name not in records or records[name].shape != shape:
            actual = records[name].shape if name in records else None
            raise ValueError(f"{name}: expected {shape}, got {actual}")
        result[name] = records[name]
    return result


def inspect(path, part, output=None, validate_payload=False):
    manifest = path.with_name("packed_manifest.json")
    with path.open("rb") as stream:
        if manifest.is_file():
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
            if (metadata.get("format") != FORMAT or metadata.get("base_model") != "nvidia/parakeet-tdt-0.6b-v3"
                    or metadata.get("container_sha256") != hashlib.file_digest(stream, "sha256").hexdigest()):
                raise ValueError("Phonon manifest or model SHA-256 mismatch")
            stream.seek(0)
        all_records = index(stream)
        tdt = decoder_records(all_records)
        # These BatchNorm training counters may have overflowed when stored in FP16.
        inference = {name: record for name, record in all_records.items()
                     if not name.endswith(".num_batches_tracked")}
        records = {
            "all": inference,
            "tdt": tdt,
            "encoder": {name: record for name, record in inference.items() if name.startswith("encoder.")},
        }[part]
        if part == "encoder" and len(records) != 684:
            raise ValueError(f"Expected 684 FastConformer tensors, found {len(records)}")
        kinds = Counter(rec.kind for rec in records.values())
        sizes = {name: prod(rec.shape) * 4 for name, rec in records.items()}
        if output is not None or validate_payload:
            if output is None:
                for record in records.values():
                    if not np.isfinite(expand(stream, record)).all():
                        raise ValueError(f"Nonfinite Phonon tensor {record.name}")
            else:
                save(stream, records, sizes, output)
        return {"format": FORMAT, "part": part, "tensors": len(records), "record_kinds": dict(kinds),
                "parameters": sum(sizes.values()) // 4, "packed_bytes": path.stat().st_size,
                "manifest_checked": manifest.is_file(), "payload_checked": output is not None or validate_payload,
                "training_counters_skipped": len(all_records) - len(inference)}


def save(source, records, sizes, output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    header = {"__metadata__": {"source_format": FORMAT}}
    offset = 0
    for name, record in records.items():
        end = offset + sizes[name]
        header[name] = {"dtype": "F32", "shape": record.shape, "data_offsets": [offset, end]}
        offset = end
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    with TemporaryDirectory(prefix="phonon-prepare-", dir=output.parent) as temp:
        staged = Path(temp) / "phonon.safetensors"
        with staged.open("xb") as destination:
            destination.write(struct.pack("<Q", len(encoded)) + encoded)
            for record in records.values():
                tensor = expand(source, record)
                if not np.isfinite(tensor).all():
                    raise ValueError(f"Nonfinite Phonon tensor {record.name}")
                tensor.astype("<f4", copy=False).tofile(destination)
        output.hardlink_to(staged)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="extracted model.fermion or its directory")
    parser.add_argument("--part", choices=("all", "tdt", "encoder"), default="all")
    parser.add_argument("--validate-payload", action="store_true", help="decode and verify every selected tensor without an output")
    parser.add_argument("--output", type=Path, help="new .safetensors path")
    args = parser.parse_args()
    path = args.model.resolve(strict=True)
    if path.is_dir():
        path /= "model.fermion"
    print(json.dumps(inspect(path, args.part, args.output, args.validate_payload)))


if __name__ == "__main__":
    main()
