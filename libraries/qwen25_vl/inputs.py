"""Pack Qwen2.5-VL patch and token inputs for the native encoder."""

import numpy as np


def vision_records(patches, grid_thw):
    grids = np.asarray(grid_thw)
    if (grids.ndim != 2 or grids.shape[1] != 3 or not len(grids) or not np.issubdtype(grids.dtype, np.integer)
            or np.any(grids < 1) or np.any(grids[:, 1:] % 2)):
        raise ValueError("Use one or more integer (time, height, width) grids with even spatial dimensions")
    count = sum(int(t) * int(h) * int(w) for t, h, w in grids)
    patches = np.asarray(patches, dtype="<f4")
    if count > 2048 or patches.shape != (count, 1176):
        raise ValueError("Expected at most 2048 Qwen2.5-VL patches with 1176 values each")
    records = np.empty((count, 1180), dtype="<f4")
    records[:, 4:] = patches
    offset = window_base = frame_base = 0
    for time, height, width in grids.tolist():
        frame, row, col, dy, dx = np.indices((time, height // 2, width // 2, 2, 2))
        windows_h, windows_w = (height + 7) // 8, (width + 7) // 8
        windows_per_frame = windows_h * windows_w
        window = window_base + frame * windows_per_frame + (row // 4) * windows_w + col // 4
        size = time * height * width
        records[offset:offset + size, :4] = np.stack((row * 2 + dy, col * 2 + dx, window, frame_base + frame), axis=-1).reshape(-1, 4)
        offset += size
        frame_base += time
        window_base += time * windows_per_frame
    return records


def text_records(token_ids, width=3584, *, positions=None, attention_mask=None, embeddings=None):
    ids = np.asarray(token_ids)
    if ids.ndim != 1 or not 0 < len(ids) <= 512 or not np.issubdtype(ids.dtype, np.integer) or (ids < 0).any():
        raise ValueError("Expected 1 to 512 nonnegative integer token IDs")
    count = len(ids)
    positions = np.repeat(np.arange(count)[:, None], 3, axis=1) if positions is None else np.asarray(positions)
    attention_mask = np.ones(count) if attention_mask is None else np.asarray(attention_mask)
    if positions.shape != (count, 3) or attention_mask.shape != (count,):
        raise ValueError("Expected one (time, row, column) position and one mask value per token")
    data = np.zeros((count, width + 5), dtype="<f4")
    data[:, 0], data[:, 1:4], data[:, 4] = ids, positions, attention_mask
    for index, value in (embeddings or {}).items():
        value = np.asarray(value)
        if not 0 <= index < count or value.shape != (width,):
            raise ValueError("A replacement embedding must match one token and the model width")
        data[index, 0] = -1
        data[index, 5:] = value
    return data


def image_positions(prefix, height, width, suffix):
    """MRoPE positions for text with one image grid after spatial merging."""
    before = np.repeat(np.arange(prefix)[:, None], 3, axis=1)
    row, col = np.indices((height, width))
    image = prefix + np.stack((np.zeros_like(row), row, col), axis=-1).reshape(-1, 3)
    start = prefix + max(height, width)
    after = np.repeat(np.arange(start, start + suffix)[:, None], 3, axis=1)
    return np.concatenate((before, image, after))
