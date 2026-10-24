"""
baw_core/detector.py  —  BiochemicalTransformerDetector
--------------------------------------------------------
A small Transformer that reads a [B, T, 4] packet and returns a
single integrity score in [0, 1].

ARCHITECTURE
  1. input_proj    : Linear(4, d_model)   — lifts raw features into d_model space
  2. transformer   : TransformerEncoder   — self-attention across the time axis
  3. classifier    : Linear → ReLU → Linear → Sigmoid

CAVEAT (important to understand)
  This model is UNTRAINED. Weights are random, so verify_audio_integrity()
  currently returns a cosine-similarity between the original packet and the
  signal extracted from the watermarked file.  That cosine similarity will
  be meaningful once the embedder buries the features faithfully — even
  without training the classifier head.

  Full training (pairs of genuine vs tampered packets) is the next milestone
  once the encode/decode pipeline is proven end-to-end.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np


class BiochemicalTransformerDetector(nn.Module):

    def __init__(
        self,
        feature_dim: int = 4,   # must match parser's output — pitch, volume, bond_dis, chiral
        d_model:     int = 32,
        nhead:       int = 4,
        num_layers:  int = 2,
    ):
        super().__init__()

        # ── 1. project raw 4-feature vectors into transformer space ──
        self.input_proj = nn.Linear(feature_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ── 2. classification head (will be useful once trained) ──
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------ #
    #  forward — raw tensor → scalar score                                #
    # ------------------------------------------------------------------ #
    def forward(self, packet: torch.Tensor) -> torch.Tensor:
        """
        packet : [B, T, 4]  (feature_dim must match __init__)
        returns: [B, 1]     score in (0, 1)
        """
        x      = self.input_proj(packet)   # [B, T, d_model]
        x      = self.transformer(x)       # [B, T, d_model]
        pooled = x.mean(dim=1)             # [B, d_model]  — global avg pooling
        return self.classifier(pooled)     # [B, 1]

    # ------------------------------------------------------------------ #
    #  _encode — return pooled transformer repr (before classifier head)  #
    # ------------------------------------------------------------------ #
    def _encode(self, packet: torch.Tensor) -> torch.Tensor:
        """Returns the pooled latent vector for a packet. Shape: [B, d_model]."""
        x = self.input_proj(packet)
        x = self.transformer(x)
        return x.mean(dim=1)

    # ------------------------------------------------------------------ #
    #  _extract_from_audio — pull embedded signature back out of a .wav   #
    # ------------------------------------------------------------------ #
    def _extract_from_audio(
        self,
        audio_path:   str,
        target_shape: tuple,          # (B, T, F) from the original packet
    ) -> torch.Tensor:
        """
        Reads the watermarked audio and reconstructs a [B, T, F] tensor
        by isolating the high-frequency band where the embedder wrote the
        signature and projecting it back to F feature channels.

        NOTE: this will only recover a meaningful signal if the embedder
        placed its data in the top half of the STFT spectrum. If your
        embedder uses a different band, adjust `high_freq_bins` below.
        """
        B, T, F = target_shape

        # 1. load audio
        audio, sr = librosa.load(audio_path, sr=None, mono=True)

        # 2. STFT — same n_fft / hop as embedder (adjust if yours differs)
        stft = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))

        # 3. isolate top-half frequency bins (where signature lives)
        high_freq_bins = stft[stft.shape[0] // 2 :, :]  # [freq_bins//2, frames]
        frames = high_freq_bins.T                        # [frames, freq_bins//2]

        # 4. trim / pad to exactly T time steps
        if frames.shape[0] >= T:
            frames = frames[:T, :]
        else:
            pad_len = T - frames.shape[0]
            frames  = np.pad(frames, ((0, pad_len), (0, 0)))

        # 5. project down to F feature channels by averaging chunks
        bins_per_feat = frames.shape[1] // F
        features = np.stack(
            [frames[:, i * bins_per_feat : (i + 1) * bins_per_feat].mean(axis=1)
             for i in range(F)],
            axis=-1,
        )  # [T, F]

        # 6. z-score normalise so scale differences don't kill cosine sim
        features = (features - features.mean()) / (features.std() + 1e-8)

        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # [1, T, F]

    # ------------------------------------------------------------------ #
    #  verify_audio_integrity — the main public API                       #
    # ------------------------------------------------------------------ #
    def verify_audio_integrity(
        self,
        watermarked_path:  str,
        original_packet:   torch.Tensor,
    ) -> float:
        """
        Extracts the chemical-acoustic signature from `watermarked_path`
        and compares it to `original_packet` using cosine similarity in
        the transformer's latent space.

        Returns
        -------
        float in [-1, 1]:
            ~  1.0  →  signature intact      (genuine)
            ~  0.0  →  uncorrelated          (borderline)
            ~ -1.0  →  signature destroyed   (cloned / tampered)

        Usage
        -----
        packet   = parser.build_packet("your_audio.wav", "C8H10N4O2", sequence_length=50)
        embedder.embed_signature("your_audio.wav", "watermarked.wav", packet)
        score    = detector.verify_audio_integrity("watermarked.wav", packet)
        print(score)   # should be close to 1.0 for an intact watermark
        """
        self.eval()
        with torch.no_grad():

            # --- encode the ground-truth packet -------------------------
            orig_repr = self._encode(original_packet)          # [1, d_model]

            # --- extract signature from watermarked audio ---------------
            extracted = self._extract_from_audio(
                watermarked_path,
                original_packet.shape,
            )                                                  # [1, T, F]

            extracted_repr = self._encode(extracted)           # [1, d_model]

            # --- cosine similarity between the two representations ------
            score = F.cosine_similarity(orig_repr, extracted_repr, dim=-1)

        return float(score.mean().item())
