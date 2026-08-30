import os
from flask import render_template, jsonify, request
from app import app

from baw_core.parser import ChemistryMusicParser
from baw_core.embedder import BiochemicalEmbedder
from baw_core.detector import BiochemicalTransformerDetector

parser = ChemistryMusicParser()
embedder = BiochemicalEmbedder()
detector = BiochemicalTransformerDetector()

SEQUENCE_LENGTH = 50  # time steps per audio clip; must match on embed AND verify


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

    temp_input = "temp_raw_voice.wav"
    temp_output = "protected_voice_output.wav"
    audio_file.save(temp_input)

    try:
        # Build the [1, T, 4] tensor: [pitch, volume, bond_dissonance, chirality]
        packet = parser.build_packet(temp_input, molecule, sequence_length=SEQUENCE_LENGTH)

        # Embed it into the waveform as near-inaudible carrier signals
        embedder.embed_signature(temp_input, temp_output, packet)

        # Integrity score right after embedding should be high -- this is
        # a sanity check, not yet the "was this file tampered with" check
        # (see /api/verify for that; and remember detector.py is untrained)
        integrity_score = detector.verify_audio_integrity(packet)

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
    temp_path = "temp_verify_voice.wav"
    audio_file.save(temp_path)

    try:
        recovered_packet = embedder.extract_signature(temp_path, sequence_length=SEQUENCE_LENGTH)
        integrity_score = detector.verify_audio_integrity(recovered_packet)

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify({
            "status": "success",
            "integrity_rating": f"{integrity_score * 100:.2f}%",
            "note": "detector.py is untrained -- this score is not yet meaningful.",
        })

    except Exception as e:
        return jsonify({"status": "error", "msg": f"Verification pipeline error: {str(e)}"}), 500
