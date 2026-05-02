#!/usr/bin/env bash
# Bootstrap a Jetson Orin Nano 8GB as a line edge box.
#
# Tested on: JetPack 6.x (Ubuntu 22.04, CUDA 12.2, ARM64).
#
# What this installs:
#   - PyTorch (NVIDIA-built ARM wheel for Jetson)
#   - aoi-sentinel with edge-only extras (no train deps)
#   - Lightweight image encoder (MobileNetV3 / ConvNeXt-Pico)
#   - Edge daemon + operator UI
#
# What is NOT installed:
#   - mamba-ssm / causal-conv1d  → training stays on the trainer box
#   - heavy training extras
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/aoi-sentinel}"

echo "[1/5] system packages"
sudo apt update
sudo apt install -y build-essential git curl python3 python3-venv python3-pip \
                    libgl1 libglib2.0-0 sqlite3 nginx

echo "[2/5] clone or update repo"
if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone https://github.com/DrJinHoChoi/aoi-sentinel.git "${REPO_DIR}"
fi
cd "${REPO_DIR}"
git pull

echo "[3/5] virtualenv"
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel

echo "[4/5] PyTorch for Jetson (NVIDIA ARM wheel)"
# JetPack 6 ships a system PyTorch; otherwise fetch from NVIDIA's pip index.
pip install --extra-index-url https://pypi.nvidia.com torch torchvision || \
  python3 -c "import torch" || {
    echo "fetch JetPack-matched torch wheel manually from"
    echo "  https://forums.developer.nvidia.com/t/pytorch-for-jetson/"
    exit 1
  }

echo "[5/5] aoi-sentinel (edge profile)"
pip install -e "."
pip install timm fastapi uvicorn[standard] httpx pydantic pyyaml pillow opencv-python-headless \
            jinja2 python-multipart

echo "done. start the edge daemon with:"
echo "  source .venv/bin/activate && python -m aoi_sentinel.runtime.edge --config configs/edge.yaml"
