"""Check the placement reported by native probes after inspecting their output buffers."""

import re


PLATFORMS = {"cpu": "cpu", "cuda": "iree_cuda", "rocm": "iree_rocm", "vulkan": "iree_vulkan"}


def verify_placement(stderr, backend, repeats, expected_device=None):
    placements = re.findall(r"\bbackend=(\w+) pjrt=(\w+) device=(.+?) shards=(\d+)$", stderr, re.MULTILINE)
    if len(placements) != repeats:
        raise RuntimeError(f"Expected {repeats} native placement reports, got {placements}")
    for actual, platform, device, shards in placements:
        if (actual, platform, shards) != (backend, PLATFORMS[backend], "1"):
            raise RuntimeError(f"Unexpected output placement {actual}/{platform}/{device} ({shards} shards)")
        if backend == "cpu" and device != "cpu:0":
            raise RuntimeError(f"Expected cpu:0, got {device}")
        if expected_device and expected_device.casefold() not in device.casefold():
            raise RuntimeError(f"Expected device matching {expected_device!r}, got {device!r}")
    if placements:
        actual, platform, device, shards = placements[-1]
        return {"backend": actual, "pjrt": platform, "device": device, "shards": int(shards)}
