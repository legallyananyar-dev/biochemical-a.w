"""
detector.py
-----------
BiochemicalTransformerDetector: a small Transformer that reads a
[1, T, 4] packet and outputs a single "integrity score" in [0, 1].

Conceptually:
  - Self-attention lets every time step "look at" every other time
    step, learning what a normal, untampered relationship between
    pitch/volume/dissonance/chirality looks like across the clip.
  - The per-step representations are pooled (averaged over time) and
    passed through a small classifier head that outputs one number:
    "how much does this still look like a genuine, unmodified packet."

IMPORTANT CAVEAT: this architecture is UNTRAINED. Right now its
weights are random, so verify_audio_integrity() will return a
meaningless (roughly random-ish, ~0.5) score until it's actually
trained on pairs of (genuine packet, tampered packet). Training data
-- pairs of watermarked audio and the same audio after being run
through real voice-conversion / source-separation tools -- is the
next real milestone once the encode/decode plumbing works end-to-end.
"""

import torch
import torch.nn as nn


class BiochemicalTransformerDetector(nn.Module):
    def __init__(self, feature_dim: int = 4, d_model: int = 32, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        # project the 4 raw features up into a higher-dimensional space
        # the transformer can work with more expressively
        self.input_proj = nn.Linear(feature_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, packet: torch.Tensor) -> torch.Tensor:
        # packet: [B, T, 4]
        x = self.input_proj(packet)      # [B, T, d_model]
        x = self.transformer(x)          # [B, T, d_model]
        pooled = x.mean(dim=1)           # [B, d_model]  average over time steps
        score = self.classifier(pooled)  # [B, 1]
        return score

    def verify_audio_integrity(self, packet: torch.Tensor) -> float:
        """Convenience wrapper matching the routes.py call signature."""
        self.eval()
        with torch.no_grad():
            score = self.forward(packet)
        return float(score.item())
