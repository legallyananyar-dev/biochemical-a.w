"""
parser.py
---------
Builds the [T, 4] feature packet: [pitch, volume, bond_dissonance, chirality].

Pure numpy/scipy (no torch, no librosa) so the whole app fits in a Vercel
function. Pitch uses a simple autocorrelation tracker instead of librosa.pyin.
"""

import io

import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt, resample_poly

from .chem_features import get_molecule_features

FEATURE_ORDER = ["pitch", "volume", "bond_dissonance", "chirality"]
MIN_SR = 44100  # carriers sit at 18-19.5 kHz; below this they would alias into the audible band


def load_audio(data: bytes):
    """Decode bytes -> (mono float32 array, sample rate), upsampled to >= 44.1 kHz."""
    y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if len(y) == 0:
        raise ValueError("Audio file appears to be empty")
    if sr < MIN_SR:
        y = resample_poly(y, MIN_SR, sr).astype(np.float32)
        sr = MIN_SR
    return y, sr


def tile_to_length(values: np.ndarray, length: int, smooth_window: int = 3) -> np.ndarray:
    """Repeat `values` end-to-end to `length` steps, lightly smoothing the seams.
    Tiling (not stretching) keeps small discontinuities that a denoiser would erase."""
    if len(values) == 0:
        raise ValueError("Cannot tile an empty feature array")
    tiled = np.tile(values, int(np.ceil(length / len(values))))[:length]
    if smooth_window > 1:
        tiled = np.convolve(tiled, np.ones(smooth_window) / smooth_window, mode="same")
    return tiled.astype(np.float32)


def _normalize(a: np.ndarray) -> np.ndarray:
    rng = a.max() - a.min()
    return ((a - a.min()) / rng).astype(np.float32) if rng > 0 else np.zeros_like(a, dtype=np.float32)


def _f0(frame: np.ndarray, sr: int, fmin=65.0, fmax=1000.0) -> float:
    """Autocorrelation pitch estimate in Hz; 0.0 when unvoiced."""
    frame = frame - frame.mean()
    if np.sqrt(np.mean(frame ** 2)) < 1e-4:
        return 0.0
    n = len(frame)
    ac = np.fft.irfft(np.abs(np.fft.rfft(frame, 2 * n)) ** 2)[:n]
    ac = ac / ac[0]
    lo, hi = int(sr / fmax), min(int(sr / fmin), n - 1)
    if hi <= lo:
        return 0.0
    k = lo + int(np.argmax(ac[lo:hi]))
    return sr / k if ac[k] > 0.3 else 0.0


def extract_audio_features(y: np.ndarray, sr: int, n_steps: int):
    """(pitch, volume), each length n_steps, normalized to [0, 1].
    Audio is low-passed at 5 kHz first so the 18 kHz+ watermark never feeds back into them."""
    b, a = butter(4, 5000 / (sr / 2))
    y = filtfilt(b, a, y).astype(np.float32)
    step = len(y) // n_steps
    if step < 1:
        raise ValueError("Audio is too short")
    win = min(step, int(0.08 * sr))
    pitch, volume = np.zeros(n_steps), np.zeros(n_steps)
    for t in range(n_steps):
        seg = y[t * step:(t + 1) * step if t < n_steps - 1 else len(y)]
        volume[t] = np.sqrt(np.mean(seg ** 2))
        mid = len(seg) // 2
        pitch[t] = _f0(seg[max(0, mid - win // 2): mid + win // 2], sr)
    return _normalize(pitch), _normalize(volume)


def chemical_tracks(molecule: str, n_steps: int):
    dissonance, chirality = get_molecule_features(molecule)
    return tile_to_length(dissonance, n_steps), tile_to_length(chirality, n_steps)


def build_packet(y: np.ndarray, sr: int, molecule: str, n_steps: int = 50) -> np.ndarray:
    """audio + molecule -> packet [T, 4] (order: FEATURE_ORDER)."""
    pitch, volume = extract_audio_features(y, sr, n_steps)
    dissonance, chirality = chemical_tracks(molecule, n_steps)
    return np.stack([pitch, volume, dissonance, chirality], axis=-1).astype(np.float32)
