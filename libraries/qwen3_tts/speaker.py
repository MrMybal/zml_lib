"""Host audio preprocessing for the Qwen3-TTS Base speaker encoder."""

from functools import lru_cache

import numpy as np


RATE = 24_000
FFT = 1024
HOP = 256
MELS = 128
MIN_FRAMES = 5
MAX_SECONDS = 30


def read_reference(path):
    import librosa
    import soundfile

    with soundfile.SoundFile(path) as audio:
        if not 0 < len(audio) <= MAX_SECONDS * audio.samplerate:
            raise ValueError(f"Use a reference recording at most {MAX_SECONDS} seconds long")
        rate = audio.samplerate
        samples = audio.read(dtype="float32", always_2d=True).mean(axis=1)
    if not np.isfinite(samples).all():
        raise ValueError("The reference audio contains nonfinite samples")
    if rate != RATE:
        samples = librosa.resample(samples, orig_sr=rate, target_sr=RATE)
    if not MIN_FRAMES * HOP <= len(samples) <= MAX_SECONDS * RATE:
        raise ValueError(f"Use at least {MIN_FRAMES} log-mel frames and at most {MAX_SECONDS} seconds of reference audio")
    return samples


@lru_cache(maxsize=1)
def filters():
    from librosa.filters import mel
    from scipy.signal.windows import hann

    return hann(FFT, sym=False).astype(np.float32), mel(sr=RATE, n_fft=FFT, n_mels=MELS, fmin=0, fmax=RATE // 2)


def log_mels(samples):
    """Qwen speaker log-mels in time-major order, with no utterance normalization."""
    from scipy.fft import rfft

    pad = (FFT - HOP) // 2
    padded = np.pad(samples, (pad, pad), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, FFT)[::HOP]
    window, mel = filters()
    spectrum = rfft(frames * window, axis=-1)
    magnitude = np.sqrt(spectrum.real ** 2 + spectrum.imag ** 2 + np.float32(1e-9))
    return np.log(np.maximum(magnitude @ mel.T, np.float32(1e-5)))
