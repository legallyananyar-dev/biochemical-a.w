# Biochemical Acoustic Watermarking (BAW) 🧬🎵

An advanced zero-trust cybersecurity framework that maps strict, non-linear geometric properties of chemical compounds onto multi-dimensional audio packets to protect human voices from unauthorized AI voice cloning.

## 🚀 How It Works
1. **Chemical Key Mapping:** A chosen molecule is parsed into specific audio parameters: `[Pitch, Volume, Bond Dissonance, Chiral Filter]`.
2. **Steganographic Injection:** The parameters are embedded into your voice file as high-frequency, low-amplitude masking data (perceived as faint white noise).
3. **Transformer Sentinels:** A neural network continuously checks if the audio's physical sound parameters perfectly align with the underlying molecular physics rules.
4. **Clone Tripwire:** If an external AI system tries to isolate, scrape, or compress your voice, the delicate biochemical data pattern collapses, triggering a security alert.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py            # http://127.0.0.1:5000
python tests/selfcheck.py
```

## Deploy to Vercel
1. Push this repo to GitHub, then **Add New > Project** in Vercel and import it. No build command or output directory needed (Vercel detects Flask from `app.py`).
2. Or from the CLI: `npm i -g vercel && vercel --prod`.

Layout: `app.py` (Flask entrypoint) / `routes.py` (`/api/config`, `/api/key`, `/api/protect`, `/api/verify`) / `public/` (UI, served by the CDN) / `baw_core/` (signal + chemistry code) / `vercel.json`.

**Hosted-demo limits:** uploads up to 4 MB and clips up to about 49 s (Vercel caps request and response bodies at 4.5 MB). For longer audio, move uploads to Vercel Blob.

**Deploy dependencies are slim on purpose.** `requirements.txt` has no torch or librosa (too big for a function). `baw_core/detector.py`, the notebook and `requirements-research.txt` are research-only and excluded from the deploy.

## How verification scores
Recovered carrier tracks are compared with the tracks the molecule and the audio itself predict (Pearson r), then multiplied by how much carrier energy is left. Filtering or denoising the high band drops the score even when the pattern shape survives.
