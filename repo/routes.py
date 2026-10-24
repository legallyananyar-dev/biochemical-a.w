import json
import os

import numpy as np
from flask import Response, jsonify, request
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from app import app
from baw_core.chem_features import PRESETS
from baw_core.embedder import BiochemicalEmbedder, to_wav_bytes
from baw_core.parser import FEATURE_ORDER, build_packet, chemical_tracks, load_audio
from baw_core.verify import verify_audio

SEQUENCE_LENGTH = int(os.environ.get("SEQUENCE_LENGTH", 50))
SECURITY_THRESHOLD = float(os.environ.get("SECURITY_THRESHOLD", 0.75))
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", 4))        # Vercel rejects request bodies > 4.5 MB
MAX_OUT_BYTES = int(4.2 * 1024 * 1024)                           # ...and responses > 4.5 MB
embedder = BiochemicalEmbedder()


class ApiError(Exception):
    def __init__(self, msg, status=400):
        self.msg, self.status = msg, status


@app.errorhandler(ApiError)
def _api_error(e):
    return jsonify(status="error", msg=e.msg), e.status


@app.errorhandler(RequestEntityTooLarge)
def _too_large(_):
    return jsonify(status="error", msg=f"File too large. Maximum upload is {MAX_UPLOAD_MB:g} MB."), 413


@app.after_request
def _no_store(resp):
    if request.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/")
def index():  # on Vercel public/index.html is served from the CDN before this runs
    return app.send_static_file("index.html")


@app.get("/api/config")
def config():
    return jsonify(
        molecules=[{"id": k, "name": v[0], "risk": v[2]} for k, v in PRESETS.items()],
        threshold=SECURITY_THRESHOLD, max_upload_mb=MAX_UPLOAD_MB,
        steps=SEQUENCE_LENGTH, features=FEATURE_ORDER,
    )


@app.get("/api/key")
def key():
    """Chemical half of the key for a molecule (used by the UI preview)."""
    mol = _molecule(request.args.get("molecule"))
    try:
        d, c = chemical_tracks(mol, SEQUENCE_LENGTH)
    except ValueError as e:
        raise ApiError(str(e), 422)
    return jsonify(status="success", dissonance=_r(d), chirality=_r(c))


def _r(a):
    return np.round(np.asarray(a, dtype=float), 3).tolist()


def _molecule(value):
    mol = (value or "H2O").strip()
    if not mol or len(mol) > 200:
        raise ApiError("Molecule must be a preset id or a SMILES string up to 200 characters.")
    return mol


def _read_audio():
    f = request.files.get("audio")
    if f is None or not f.filename:
        raise ApiError("No audio file uploaded.")
    data = f.read(int(MAX_UPLOAD_MB * 1024 * 1024) + 1)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ApiError(f"File too large. Maximum upload is {MAX_UPLOAD_MB:g} MB.", 413)
    try:
        y, sr = load_audio(data)
    except Exception:
        raise ApiError("Could not read that audio file. Use WAV, FLAC, OGG or MP3.", 415)
    if len(y) < SEQUENCE_LENGTH * 64:
        raise ApiError("Audio is too short. Use at least a few seconds of speech.", 422)
    return secure_filename(os.path.splitext(f.filename)[0]) or "voice", y, sr


def _report(res):
    return {"score": round(res["score"], 4), "shape": round(res["shape"], 4), "level": round(res["level"], 4),
            "tracks": {k: (None if v is None else round(v, 4)) for k, v in res["tracks"].items()},
            "expected": _r(res["expected"]), "recovered": _r(res["recovered"])}


@app.post("/api/protect")
def protect():
    molecule = _molecule(request.form.get("molecule"))
    name, y, sr = _read_audio()
    if len(y) * 2 + 44 > MAX_OUT_BYTES:
        raise ApiError(f"Clip too long for the hosted demo ({len(y) / sr:.0f}s). Use up to "
                       f"{int(MAX_OUT_BYTES / (sr * 2))}s at {sr / 1000:g} kHz.", 413)
    try:
        packet = build_packet(y, sr, molecule, SEQUENCE_LENGTH)
        wav = to_wav_bytes(embedder.embed(y, sr, packet), sr)
        y2, sr2 = load_audio(wav)  # read the file back, exactly as a verifier would receive it
        report = _report(verify_audio(embedder, y2, sr2, molecule, SEQUENCE_LENGTH))
    except ValueError as e:
        raise ApiError(str(e), 422)
    report.update(molecule=molecule, sample_rate=sr, seconds=round(len(y) / sr, 2))
    return Response(wav, mimetype="audio/wav", headers={
        "Content-Disposition": f'attachment; filename="baw_{name}.wav"',
        "X-BAW-Result": json.dumps(report, separators=(",", ":")),
    })


@app.post("/api/verify")
def verify():
    """molecule = preset id / SMILES to test against, or 'auto' (default) to try every preset."""
    choice = (request.form.get("molecule") or "auto").strip()
    _, y, sr = _read_audio()
    candidates = list(PRESETS) if choice == "auto" else [_molecule(choice)]
    try:
        results = sorted(((m, verify_audio(embedder, y, sr, m, SEQUENCE_LENGTH)) for m in candidates),
                         key=lambda kv: kv[1]["score"], reverse=True)
    except ValueError as e:
        raise ApiError(str(e), 422)
    best_id, best = results[0]
    verdict = "intact" if best["score"] >= SECURITY_THRESHOLD else ("absent" if best["level"] < 0.15 else "damaged")
    return jsonify(status="success", verdict=verdict, threshold=SECURITY_THRESHOLD, molecule=best_id,
                   candidates=[{"id": m, "score": round(r["score"], 4)} for m, r in results],
                   **_report(best))
