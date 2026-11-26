"""Pack image latents and conditioning for the native Qwen Image Edit probe."""

import numpy as np


def pack_latent(latent):
    array = np.asarray(latent, dtype="<f4")
    if array.ndim != 3 or array.shape[0] != 16 or any(size == 0 or size % 2 for size in array.shape[1:]):
        raise ValueError("Expected a planar [16,H,W] latent with positive even spatial dimensions")
    _, height, width = array.shape
    return array.reshape(16, height // 2, 2, width // 2, 2).transpose(1, 3, 0, 2, 4).reshape(-1, 64)


def unpack_latent(patches, height, width):
    return np.asarray(patches, dtype="<f4").reshape(height // 2, width // 2, 16, 2, 2).transpose(2, 0, 3, 1, 4).reshape(16, height, width)


def positions(height, width, image_index):
    rows, cols = height // 2, width // 2
    row, col = np.indices((rows, cols))
    return np.stack((np.full_like(row, image_index), row - rows // 2, col - cols // 2), axis=-1).reshape(-1, 3)


def write_denoise_input(output, target, references, text, mask, times, *, reference_states=None):
    images = [target, *references]
    patches = [pack_latent(image) for image in images]
    coords = [positions(*image.shape[1:], index) for index, image in enumerate(images)]
    text = np.asarray(text, dtype="<f4")
    mask = np.asarray(mask, dtype="<f4")
    times = np.asarray(times, dtype="<f4")
    if text.ndim != 2 or len(text) == 0 or mask.shape != (len(text),) or times.ndim != 1 or not len(times):
        raise ValueError("Expected a text sequence, its mask and one or more timesteps")
    states = ()
    if reference_states is not None:
        if len(times) < 2 or len(reference_states) != len(times) - 2:
            raise ValueError("Expected one extra reference state per Euler step after the first")
        if any(np.shape(state) != np.shape(target) for state in reference_states):
            raise ValueError("Each reference state must use the target latent shape")
        states = [pack_latent(state) for state in reference_states]
    text_start = max(target.shape[1:]) // 4
    text_coords = np.repeat(np.arange(text_start, text_start + len(text))[:, None], 3, axis=1)
    with output.open("xb") as stream:
        for part in (*patches, *coords, text, text_coords, mask, times, *states):
            np.asarray(part, dtype="<f4", order="C").tofile(stream)
    return len(patches[0]), sum(map(len, patches)), len(text), len(times) - 1
