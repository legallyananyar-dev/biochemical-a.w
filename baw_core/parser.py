"""
parser.py
---------
ChemistryMusicParser: combines per-time-step AUDIO features (pitch,
volume) with per-atom CHEMICAL features (bond dissonance, chirality)
into a single PyTorch tensor shaped [1, T, 4].

The molecule has a fixed number of atoms; the audio has a fixed
number of time steps (T, e.g. 50). These two counts almost never
match, so `tile_to_length` repeats the chemical arrays to cover T.
"""

import numpy as np
import torch
import librosa

from .chem_features import get_molecule_features

FEATURE_ORDER = ["pitch", "volume", "bond_dissonance", "chirality"]


def tile_to_length(values: np.ndarray, length: int, smooth_window: int = 3) -> np.ndarray:
    """
    Repeat `values` end-to-end until it covers `length` steps, then
    lightly smooth the seams where it wraps around.

    Tiling (rather than smoothly stretching) is a deliberate choice:
    it keeps small discontinuities in the chemical feature track. That
    makes the pattern easier for the transformer to learn as "expected
    structure," and easier to notice breaking when something (like an
    attacker's denoiser) smooths it away -- which is exactly the
    tamper signal this project is built around.
    """
    n = len(values)
    if n == 0:
        raise ValueError("Cannot tile an empty feature array")

    reps = int(np.ceil(length / n))
    tiled = np.tile(values, reps)[:length]

    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        tiled = np.convolve(tiled, kernel, mode="same")

    return tiled.astype(np.float32)


def _normalize(a: np.ndarray) -> np.ndarray:
    rng = a.max() - a.min()
    return (a - a.min()) / rng if rng > 0 else np.zeros_like(a)


def extract_audio_features(audio_path: str, n_steps: int):
    """
    Load audio and reduce it to `n_steps` (pitch, volume) pairs.
      - volume: RMS energy per step
      - pitch:  estimated fundamental frequency (Hz) via librosa.pyin
    Both are resampled to exactly n_steps points and normalized to [0, 1].
    """
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    if len(y) == 0:
        raise ValueError("Audio file appears to be empty")

    hop_length = max(1, len(y) // n_steps)
    rms = librosa.feature.rms(y=y, frame_length=hop_length * 2, hop_length=hop_length)[0]

    f0, _voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
        sr=sr, hop_length=hop_length
    )
    f0 = np.nan_to_num(f0, nan=0.0)  # unvoiced frames -> 0 Hz

    def resample(arr):
        if len(arr) < 2:
            return np.full(n_steps, arr[0] if len(arr) else 0.0, dtype=np.float32)
        x_old = np.linspace(0, 1, len(arr))
        x_new = np.linspace(0, 1, n_steps)
        return np.interp(x_new, x_old, arr).astype(np.float32)

    return _normalize(resample(f0)), _normalize(resample(rms))


class ChemistryMusicParser:
    def get_chemical_track(self, molecule_input: str, sequence_length: int):
        """molecule string -> (dissonance_track, chirality_track), each length T."""
        dissonance, chirality = get_molecule_features(molecule_input)
        return (
            tile_to_length(dissonance, sequence_length),
            tile_to_length(chirality, sequence_length),
        )

    def build_packet(self, audio_path: str, molecule_input: str, sequence_length: int = 50) -> torch.Tensor:
        """
        Full pipeline: audio file + molecule -> tensor [1, T, 4]
        Feature order: [pitch, volume, bond_dissonance, chirality]
        """
        pitch, volume = extract_audio_features(audio_path, sequence_length)
        dissonance, chirality = self.get_chemical_track(molecule_input, sequence_length)

        combined = np.stack([pitch, volume, dissonance, chirality], axis=-1)  # [T, 4]
        tensor = torch.tensor(combined, dtype=torch.float32).unsqueeze(0)     # [1, T, 4]
        return tensor

    # Backward-compatible alias for the older routes.py call signature
    def create_molecular_packet(self, recipe_or_molecule, sequence_length: int = 50):
        """
        Deprecated shim: the old routes.py passed a fake per-character
        'recipe' list. This now expects a molecule string (formula
        preset or SMILES) directly -- update the caller instead of
        relying on this shim long-term.
        """
        if isinstance(recipe_or_molecule, str):
            return self.get_chemical_track(recipe_or_molecule, sequence_length)
        raise TypeError(
            "create_molecular_packet now expects a molecule string (formula or SMILES), "
            "not a recipe list. Use build_packet(audio_path, molecule_str) instead."
        )
