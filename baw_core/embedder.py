"""
embedder.py
-----------
BiochemicalEmbedder: writes a [1, T, 4] packet into an audio file as a
near-inaudible high-frequency signal, and can read it back out.

METHOD (deliberately simple for a working v1 -- this is NOT yet
competing with research-grade audio watermarking like AudioSeal or
WavMark, which use trained neural encoders and survive real-world
compression/attacks. This is a hand-designed signal-processing
scheme so you have something that actually runs end to end first):

For each of the 4 features, pick a distinct carrier frequency up in
the 18-20kHz range (most adult listeners can't hear much above
~18-19kHz -- this is *why* it reads as faint white noise to a person
but is a clean, extractable pattern to software). For each time step,
splice in a burst of that carrier with amplitude proportional to the
feature's value at that step. Extraction reverses this: band-pass
filter around each carrier frequency, measure the amplitude in each
time-step window, and that recovers the approximate original values.

That extraction path is what lets detector.py be run at VERIFY time
on a file you didn't watermark yourself -- re-derive the packet from
the (possibly tampered) audio, rather than reusing the original tensor.

CAVEATS:
  - This scheme is amplitude-based and will NOT survive MP3/lossy
    compression, resampling, or real denoising the way a trained
    watermark would -- expect it to be fragile against almost
    everything right now, not just malicious tampering. Tightening
    that gap (robust to benign processing, fragile only to malicious
    processing) is real future work, same as in AudioSeal/StreamMark.
  - `amplitude` needs tuning per recording: too quiet and it's lost
    under the audio's own noise floor, too loud and it becomes audible.
"""

import numpy as np
import torch
import torchaudio
from scipy.signal import butter, filtfilt

CARRIER_BASE_HZ = 18000
CARRIER_SPACING_HZ = 500


class BiochemicalEmbedder:
    def __init__(self, amplitude: float = 0.02):
        self.amplitude = amplitude  # relative to the audio's own peak amplitude

    def embed_signature(self, input_path: str, output_path: str, packet: torch.Tensor):
        waveform, sr = torchaudio.load(input_path)  # [channels, samples]
        waveform = waveform.mean(dim=0)              # mix down to mono

        packet = packet.squeeze(0)  # [T, 4]
        T, n_features = packet.shape
        samples_per_step = max(1, len(waveform) // T)

        t_axis = torch.arange(len(waveform), dtype=torch.float32) / sr
        watermark = torch.zeros_like(waveform)

        for f in range(n_features):
            freq = CARRIER_BASE_HZ + f * CARRIER_SPACING_HZ
            carrier = torch.sin(2 * np.pi * freq * t_axis)

            envelope = torch.zeros_like(waveform)
            for t in range(T):
                start = t * samples_per_step
                end = start + samples_per_step if t < T - 1 else len(waveform)
                envelope[start:end] = packet[t, f].item()

            watermark += carrier * envelope

        peak = waveform.abs().max().clamp(min=1e-6)
        watermark = watermark * self.amplitude * peak

        watermarked = (waveform + watermark).clamp(-1.0, 1.0)
        torchaudio.save(output_path, watermarked.unsqueeze(0), sr)

    def extract_signature(self, audio_path: str, sequence_length: int, n_features: int = 4) -> torch.Tensor:
        """
        Reverse of embed_signature: recovers an approximate [1, T, 4]
        packet from a (possibly tampered) audio file, for verify-time use.
        """
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0).numpy()
        T = sequence_length
        samples_per_step = max(1, len(waveform) // T)

        recovered = np.zeros((T, n_features), dtype=np.float32)

        for f in range(n_features):
            freq = CARRIER_BASE_HZ + f * CARRIER_SPACING_HZ
            filtered = self._bandpass(waveform, sr, freq, bandwidth=100)

            for t in range(T):
                start = t * samples_per_step
                end = start + samples_per_step if t < T - 1 else len(waveform)
                segment = filtered[start:end]
                recovered[t, f] = np.sqrt(np.mean(segment ** 2)) if len(segment) else 0.0

        for f in range(n_features):
            col = recovered[:, f]
            rng = col.max() - col.min()
            if rng > 0:
                recovered[:, f] = (col - col.min()) / rng

        return torch.tensor(recovered, dtype=torch.float32).unsqueeze(0)

    @staticmethod
    def _bandpass(signal: np.ndarray, sr: int, center_freq: float, bandwidth: float) -> np.ndarray:
        nyquist = sr / 2
        low = max(1e-6, (center_freq - bandwidth / 2)) / nyquist
        high = min(0.999, (center_freq + bandwidth / 2) / nyquist)
        b, a = butter(4, [low, high], btype='band')
        return filtfilt(b, a, signal)
