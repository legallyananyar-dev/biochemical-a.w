"""
embedder.py
-----------
Writes a [T, 4] packet into audio as four near-inaudible carriers (18-19.5 kHz,
amplitude = feature value per time step) and reads it back with band-pass +
RMS per step.

v1 hand-designed scheme: amplitude based, so it will NOT survive MP3 / resampling /
denoising -- that fragility is what the verify step uses as its tamper signal.
"""

import io
import os

import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt

CARRIER_BASE_HZ = float(os.environ.get("CARRIER_BASE_HZ", 18000))
CARRIER_SPACING_HZ = float(os.environ.get("CARRIER_SPACING_HZ", 500))


def step_index(n_samples: int, n_steps: int) -> np.ndarray:
    """Time-step id of every sample (shared by embed and extract so they stay aligned)."""
    return np.minimum(np.arange(n_samples) * n_steps // n_samples, n_steps - 1)


class BiochemicalEmbedder:
    def __init__(self, amplitude: float = None):
        self.amplitude = float(os.environ.get("WATERMARK_AMPLITUDE", 0.02)) if amplitude is None else amplitude

    def embed(self, y: np.ndarray, sr: int, packet: np.ndarray) -> np.ndarray:
        n_steps, n_feat = packet.shape
        t = np.arange(len(y)) / sr
        idx = step_index(len(y), n_steps)
        wm = np.zeros(len(y))
        for f in range(n_feat):
            wm += np.sin(2 * np.pi * (CARRIER_BASE_HZ + f * CARRIER_SPACING_HZ) * t) * packet[idx, f]
        peak = max(float(np.abs(y).max()), 1e-6)
        return np.clip(y + wm * self.amplitude * peak, -1.0, 1.0).astype(np.float32)

    def extract_raw(self, y: np.ndarray, sr: int, n_steps: int, n_feat: int = 4) -> np.ndarray:
        """Per-step RMS of each carrier band, NOT normalized -> [T, 4] (absolute level kept)."""
        idx = step_index(len(y), n_steps)
        counts = np.maximum(np.bincount(idx, minlength=n_steps), 1)
        out = np.zeros((n_steps, n_feat), dtype=np.float32)
        for f in range(n_feat):
            band = self._bandpass(y, sr, CARRIER_BASE_HZ + f * CARRIER_SPACING_HZ, 100)
            out[:, f] = np.sqrt(np.bincount(idx, weights=band ** 2, minlength=n_steps) / counts)
        return out

    @staticmethod
    def normalize(raw: np.ndarray) -> np.ndarray:
        """Min-max each column to [0, 1] (shape only, level discarded)."""
        lo, hi = raw.min(axis=0), raw.max(axis=0)
        return np.where(hi - lo > 0, (raw - lo) / np.maximum(hi - lo, 1e-12), 0.0).astype(np.float32)

    def expected_level(self, y: np.ndarray, expected: np.ndarray) -> float:
        """Total carrier RMS an intact watermark should show for this packet (sine RMS = A/sqrt2)."""
        return float(expected.sum() * self.amplitude * max(float(np.abs(y).max()), 1e-6) / np.sqrt(2))

    @staticmethod
    def _bandpass(y, sr, center, bandwidth):
        nyq = sr / 2
        if center + bandwidth / 2 >= nyq:
            raise ValueError(f"Sample rate {sr} Hz is too low for the {center:.0f} Hz carrier")
        b, a = butter(4, [(center - bandwidth / 2) / nyq, (center + bandwidth / 2) / nyq], btype="band")
        return filtfilt(b, a, y)


def to_wav_bytes(y: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()
