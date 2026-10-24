"""Smallest check that fails if the BAW pipeline breaks: python tests/selfcheck.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.signal import butter, filtfilt
from baw_core.parser import build_packet
from baw_core.embedder import BiochemicalEmbedder, to_wav_bytes
from baw_core.parser import load_audio
from baw_core.verify import verify_audio

def voice(sr=44100, secs=3, seed=0):
    """Synthetic 'voice': gliding harmonic tone with a syllable-like amplitude envelope + a little noise."""
    rng = np.random.default_rng(seed); t = np.arange(int(sr * secs)) / sr
    f0 = 140 + 60 * np.sin(2 * np.pi * 0.7 * t)
    ph = 2 * np.pi * np.cumsum(f0) / sr
    y = sum(np.sin(k * ph) / k for k in (1, 2, 3, 4))
    y *= 0.3 + 0.7 * np.abs(np.sin(2 * np.pi * 1.3 * t))
    return (0.4 * y / np.abs(y).max() + 0.002 * rng.standard_normal(len(t))).astype(np.float32), sr

def run():
    emb, T = BiochemicalEmbedder(), 50
    y, sr = voice()
    for mol in ("H2O", "C8H10N4O2", "L-Ala"):
        marked = emb.embed(y, sr, build_packet(y, sr, mol, T))
        y2, sr2 = load_audio(to_wav_bytes(marked, sr))            # round-trip through 16-bit WAV
        good = verify_audio(emb, y2, sr2, mol, T)["score"]
        clean = verify_audio(emb, y, sr, mol, T)["score"]           # never watermarked
        b, a = butter(4, 16000 / (sr / 2)); stripped = filtfilt(b, a, y2)   # 16 kHz low-pass = "denoiser"
        tamp = verify_audio(emb, stripped.astype(np.float32), sr, mol, T)["score"]
        print(f"{mol:10s} marked={good:.2f} unmarked={clean:.2f} stripped={tamp:.2f}")
        assert good > 0.75 and clean < 0.6 and tamp < 0.6, mol
    # wrong molecule must score lower than the right one
    marked = emb.embed(y, sr, build_packet(y, sr, "C8H10N4O2", T))
    assert verify_audio(emb, marked, sr, "C8H10N4O2", T)["score"] > verify_audio(emb, marked, sr, "C9H8O4", T)["score"]
    # audible-band safety: below 12 kHz the watermark must sit >60 dB under the voice (interior RMS, ignores filter edges)
    diff = marked - y; b, a = butter(8, 12000 / (sr / 2)); low = filtfilt(b, a, diff)[sr // 10:-sr // 10]
    assert np.sqrt(np.mean(low ** 2)) < 1e-3 * np.sqrt(np.mean(y ** 2)), "watermark leaks into audible band"
    print("selfcheck OK")

if __name__ == "__main__":
    run()
