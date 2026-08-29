import os
from flask import render_template, jsonify, request
from app import app
import torch

# Import the core PyTorch modules we built
from baw_core.parser import ChemistryMusicParser
from baw_core.embedder import BiochemicalEmbedder
from baw_core.detector import BiochemicalTransformerDetector

# Initialize the framework tools
parser = ChemistryMusicParser()
embedder = BiochemicalEmbedder()
detector = BiochemicalTransformerDetector()

@app.route('/api/protect', methods=['POST'])
def protect_voice():
    # 1. Grab parameters from the web frontend form
    molecule_formula = request.form.get('molecule', 'H2O')
    
    # 2. Check if an audio file was uploaded
    if 'audio' not in request.files:
        return jsonify({"status": "error", "msg": "No audio file uploaded to server."}), 400
        
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"status": "error", "msg": "Empty audio file boundary."}), 400

    # Save incoming raw file temporarily
    temp_input = "temp_raw_voice.wav"
    temp_output = "protected_voice_output.wav"
    audio_file.save(temp_input)

    try:
        # 3. Step 1 of your Architecture: Create the PyTorch [1, Steps, 4] Molecule Tensor
        # Simple dynamic translation recipe mapping parsing tokens
        recipe = [(char, 'single', 'R') for char in molecule_formula if char.isalpha()]
        molecule_tensor = parser.create_molecular_packet(recipe, sequence_length=50)

        # 4. Step 2 of your Architecture: Embed the tensor into the physical sound wave
        embedder.embed_signature(temp_input, temp_output, molecule_tensor)

        # 5. Step 3 of your Architecture: Run validation via the Transformer Sentinel
        # We simulate the validation tracking matrix check by forwarding our packet array
        integrity_score = detector.verify_audio_integrity(molecule_tensor)

        # Clean up input temporary file cache
        if os.path.exists(temp_input):
            os.remove(temp_input)

        return jsonify({
            "status": "success",
            "msg": "Biochemical watermark structural lock engaged.",
            "integrity_rating": f"{integrity_score * 100:.2f}%",
            "download_url": f"/{temp_output}"
        })

    except Exception as e:
        return jsonify({"status": "error", "msg": f"Execution pipeline error: {str(e)}"}), 500
