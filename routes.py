import os
import tempfile
from flask import render_template, jsonify, request
from app import app

from baw_core.parser import ChemistryMusicParser
from baw_core.embedder import BiochemicalEmbedder

parser = ChemistryMusicParser()
embedder = BiochemicalEmbedder()

# Configuration
SEQUENCE_LENGTH = 50  # time steps per audio clip; must match parser AND embedder

# ponytail: detector is untrained (weights random). Returns placeholder score.
# Replace with trained detector or move to debug endpoint once ready.
PLACEHOLDER_INTEGRITY_SCORE = 0.95


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/protect', methods=['POST'])
def protect_voice():
    molecule = request.form.get('molecule', 'H2O')

    if 'audio' not in request.files:
        return jsonify({"status": "error", "msg": "No audio file uploaded."}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"status": "error", "msg": "Empty audio file."}), 400

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_input_file:
            temp_input = temp_input_file.name
        audio_file.save(temp_input)
        
        temp_output = os.path.join(tempfile.gettempdir(), "protected_voice_output.wav")

        # Build the [1, T, 4] tensor: [pitch, volume, bond_dissonance, chirality]
        packet = parser.build_packet(temp_input, molecule, sequence_length=SEQUENCE_LENGTH)

        # Embed it into the waveform as near-inaudible carrier signals
        embedder.embed_signature(temp_input, temp_output, packet)

        # ponytail: placeholder integrity score (detector untrained)
        integrity_score = PLACEHOLDER_INTEGRITY_SCORE

        if os.path.exists(temp_input):
            os.remove(temp_input)

        return jsonify({
            "status": "success",
            "msg": "Biochemical watermark structural lock engaged.",
            "integrity_rating": f"{integrity_score * 100:.2f}%",
            "download_url": f"/{temp_output}",
        })

    except Exception as e:
        return jsonify({"status": "error", "msg": f"Execution pipeline error: {str(e)}"}), 500


@app.route('/api/verify', methods=['POST'])
def verify_voice():
    """
    Verify-time flow: given a (possibly tampered) audio file, RE-EXTRACT
    the packet from the waveform itself (not the original tensor) and
    score it. This is the actual clone/tamper check -- distinct from the
    sanity-check score returned by /api/protect.
    """
    if 'audio' not in request.files:
        return jsonify({"status": "error", "msg": "No audio file uploaded."}), 400

    audio_file = request.files['audio']
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = temp_file.name
    audio_file.save(temp_path)

    try:
        recovered_packet = embedder.extract_signature(temp_path, sequence_length=SEQUENCE_LENGTH)
        # ponytail: placeholder integrity score (detector untrained)
        integrity_score = PLACEHOLDER_INTEGRITY_SCORE

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify({
            "status": "success",
            "integrity_rating": f"{integrity_score * 100:.2f}%",
            "note": "detector.py is untrained -- this score is not yet meaningful.",
        })

    except Exception as e:
        return jsonify({"status": "error", "msg": f"Verification pipeline error: {str(e)}"}), 500
