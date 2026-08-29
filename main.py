import os
import subprocess
import sys
import urllib.request

try:
    import torch
    import torchaudio
    print("✅ PyTorch environment verified successfully.")
except ImportError:
    print("📥 Bypassing index tracking. Fetching architecture binaries directly...")
    
    # 1. Direct secure paths to the stable wheel files
    torch_url = "https://pytorch.org"
    audio_url = "https://pytorch.org"
    
    torch_file = "torch_cpu.whl"
    audio_file = "audio_cpu.whl"
    
    # 2. Raw network streaming download (Render's broken index manager cannot touch this)
    print("-> Downloading core tensor wheel...")
    urllib.request.urlretrieve(torch_url, torch_file)
    
    print("-> Downloading audio architecture wheel...")
    urllib.request.urlretrieve(audio_url, audio_file)
    
    print("📦 Local downloads complete. Running standalone target installation...")
    # 3. Legally instruct pip to run locally using the saved disk files
    subprocess.check_call([sys.executable, "-m", "pip", "install", torch_file, audio_file, "--no-index"])
    
    # 4. Clean up temporary files from disk storage
    os.remove(torch_file)
    os.remove(audio_file)
    print("✅ Core architecture libraries dynamically linked.")
