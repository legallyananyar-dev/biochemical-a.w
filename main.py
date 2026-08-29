import subprocess
import sys

# Automatically run the correct fast CPU download inside Render's operating container
try:
    import torch
    import torchaudio
    print("✅ PyTorch environment verified successfully.")
except ImportError:
    print("📥 Core machine learning packages missing. Executing secure pipeline wheel download...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchaudio", "--index-url", "https://pytorch.org"])
    print("✅ Core architecture libraries dynamically linked.")
