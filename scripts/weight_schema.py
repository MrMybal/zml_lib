"""Print Zig parameter names and shapes from one or more FP32 SafeTensors headers."""

import argparse
import json
from pathlib import Path
import struct


def schema(path):
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < size <= 1024 * 1024:
            raise ValueError(f"{path}: invalid SafeTensors header length")
        header = json.loads(stream.read(size))
    for name, desc in sorted(header.items()):
        if name != "__metadata__":
            if desc["dtype"] != "F32" or not all(isinstance(d, int) and d > 0 for d in desc["shape"]):
                raise ValueError(f"{path}: unsupported tensor {name}")
            yield name, desc["shape"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path, help="file stems become Zig identifiers")
    args = parser.parse_args()
    if len(set(p.stem for p in args.checkpoints)) != len(args.checkpoints):
        parser.error("checkpoint stems must be unique")
    print('const Parameter = @import("zml_lib/runtime").layers.Parameter;')
    for path in args.checkpoints:
        if not path.stem.isascii() or not path.stem.isidentifier():
            parser.error("checkpoint stems must be ASCII identifiers")
        print(f"\npub const {path.stem} = blk: {{")
        print("    @setEvalBranchQuota(100_000);")
        print("    break :blk [_]Parameter{")
        for name, shape in schema(path):
            dims = ", ".join(map(str, shape))
            print(f"        .{{ .name = {json.dumps(name)}, .shape = .init(.{{{dims}}}, .f32) }},")
        print("    };\n};")


if __name__ == "__main__":
    main()
