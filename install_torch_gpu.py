"""
Install PyTorch with CUDA so the model runs on GPU.

Run from project root (with your venv activated):
  python install_torch_gpu.py

Or pick a CUDA version:
  python install_torch_gpu.py cu121   (CUDA 12.1, broad compatibility)
  python install_torch_gpu.py cu118   (CUDA 11.8)
  python install_torch_gpu.py cu124   (CUDA 12.4)
"""
import subprocess
import sys

# Default: CUDA 12.1 works with most recent NVIDIA drivers
CUDA_VERSIONS = {
    "cu118": "https://download.pytorch.org/whl/cu118",
    "cu121": "https://download.pytorch.org/whl/cu121",
    "cu124": "https://download.pytorch.org/whl/cu124",
    "cu126": "https://download.pytorch.org/whl/cu126",
}

def main():
    cuda = (sys.argv[1] if len(sys.argv) > 1 else "cu121").lower()
    if cuda not in CUDA_VERSIONS:
        print(f"Unknown CUDA version: {cuda}")
        print(f"Choose one of: {', '.join(CUDA_VERSIONS)}")
        sys.exit(1)
    index_url = CUDA_VERSIONS[cuda]
    print(f"Installing PyTorch with {cuda} from {index_url} ...")
    r = subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", index_url,
            "--force-reinstall",
        ],
        check=False,
    )
    if r.returncode != 0:
        sys.exit(r.returncode)
    # Verify
    import torch
    if torch.cuda.is_available():
        print(f"OK. CUDA available. Device: {torch.cuda.get_device_name(0)}")
    else:
        print("PyTorch installed but CUDA is still not available. Check your NVIDIA driver.")

if __name__ == "__main__":
    main()
