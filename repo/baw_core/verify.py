"""
verify.py
---------
Real integrity score (replaces the 0.95 placeholder): how well the packet recovered
from the audio's 18-19.5 kHz carriers matches what it SHOULD contain.

  chemical match : recovered dissonance/chirality tracks vs. the molecule's own tracks
  voice binding  : recovered pitch/volume tracks vs. pitch/volume measured from the audio now
                   (a watermark copied onto different audio will not line up)

Score = mean of the two group scores (each a mean of Pearson r clipped to [0, 1]) x carrier level,
where level = recovered carrier energy / energy an intact watermark would have, clipped to [0, 1].
Level is what catches filtering/denoising: Pearson r alone ignores how much signal is left.
Constant tracks (e.g. chirality of water) carry no information and are skipped.
Chirality sign is lost by the RMS extractor, so |chirality| is compared.
"""

import numpy as np

from .parser import FEATURE_ORDER, build_packet


def _corr(expected, recovered):
    if np.std(expected) < 1e-6:
        return None
    if np.std(recovered) < 1e-9:
        return 0.0
    return max(0.0, float(np.corrcoef(expected, recovered)[0, 1]))


def score_packet(recovered: np.ndarray, expected: np.ndarray):
    """recovered/expected: [T, 4]. Returns (score, per_track dict)."""
    exp = expected.copy()
    exp[:, 3] = np.abs(exp[:, 3])
    tracks = {name: _corr(exp[:, i], recovered[:, i]) for i, name in enumerate(FEATURE_ORDER)}
    groups = [[tracks[n] for n in names if tracks[n] is not None]
              for names in (("bond_dissonance", "chirality"), ("pitch", "volume"))]
    scores = [float(np.mean(g)) for g in groups if g]
    return (float(np.mean(scores)) if scores else 0.0), tracks


def verify_audio(embedder, y, sr, molecule: str, n_steps: int):
    """-> dict(score, shape, level, tracks, expected, recovered) for one candidate molecule."""
    raw = embedder.extract_raw(y, sr, n_steps)
    recovered = embedder.normalize(raw)
    expected = build_packet(y, sr, molecule, n_steps)
    shape, tracks = score_packet(recovered, expected)
    want = embedder.expected_level(y, expected)
    level = float(np.clip(raw.sum() / want, 0.0, 1.0)) if want > 0 else 0.0
    return {"score": shape * level, "shape": shape, "level": level, "tracks": tracks,
            "expected": expected, "recovered": recovered}
