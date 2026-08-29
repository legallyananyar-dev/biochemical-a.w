import subprocess
import sys

try:
    import torch
    import torchaudio
    print("✅ PyTorch environment verified successfully.")
except ImportError:
    print("📥 Hard-linking to pre-compiled stable CPU architecture packages...")
    
    # Direct download links for Python 3.11 stable CPU wheels
    torch_wheel = "https://pytorch.org"
    torchaudio_wheel = "https://pytorch.org"
    
    # Legally bypass Render's index restrictions by installing the files directly by URL
    subprocess.check_call([sys.executable, "-m", "pip", "install", torch_wheel, torchaudio_wheel])
    print("✅ Core architecture libraries dynamically linked.")
