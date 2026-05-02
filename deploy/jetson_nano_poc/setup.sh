#!/usr/bin/env bash
# Bootstrap a Jetson Nano (2019, 4GB, Maxwell) as the POC demo box.
#
# Tested on: JetPack 4.6.4 (Ubuntu 18.04 ARM64).
#
# Notes
# -----
# - We DO NOT install mamba-ssm / causal-conv1d. They will not compile on
#   Maxwell. The POC runs MobileNetV3-Small only.
# - Python 3.8 (not the system 3.6) gives us a usable typing/dataclass
#   surface and matches recent FastAPI / pydantic v2.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/aoi-sentinel}"
WATCH_DIR="${WATCH_DIR:-/tmp/aoi-watch}"

echo "[1/7] system packages"
sudo apt update
sudo apt install -y software-properties-common curl git build-essential

# Python 3.8 from deadsnakes (JetPack 4.6 ships 3.6.9 which is too old)
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.8 python3.8-venv python3.8-dev python3-pip \
                    libgl1 libglib2.0-0 sqlite3 nginx jq

echo "[2/7] clone or update repo"
if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone https://github.com/DrJinHoChoi/aoi-sentinel.git "${REPO_DIR}"
fi
cd "${REPO_DIR}"
git pull

echo "[3/7] virtualenv"
python3.8 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools

echo "[4/7] PyTorch for Jetson Nano (NVIDIA-built ARM wheel for JetPack 4.6)"
# The official wheel index for Jetson:
#   https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
# JetPack 4.6 supports torch 1.10 / 1.11.
TORCH_WHEEL_URL="https://nvidia.box.com/shared/static/ssf2v7pf5i245fk4i0q926hy4imzs2ph.whl"  # torch-1.11.0-cp38-cp38-linux_aarch64
TMP_WHEEL=/tmp/torch-jetson.whl
if [[ ! -f "${TMP_WHEEL}" ]]; then
  wget -O "${TMP_WHEEL}" "${TORCH_WHEEL_URL}"
fi
pip install "${TMP_WHEEL}"
pip install torchvision==0.12.0  # CPU wheel ok for the POC; vision encoder is small

echo "[5/7] aoi-sentinel (POC profile — no train/RL extras)"
pip install -e "."
pip install timm fastapi 'uvicorn[standard]' httpx pydantic pyyaml pillow \
            opencv-python-headless jinja2 python-multipart numpy pandas

echo "[6/7] watch directory + demo systemd unit"
sudo mkdir -p "${WATCH_DIR}"
sudo chown "${USER}:${USER}" "${WATCH_DIR}"

sudo cp "${REPO_DIR}/deploy/jetson_nano_poc/aoi-demo.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable aoi-demo

echo "[7/7] generate the demo bundle"
python "${REPO_DIR}/deploy/jetson_nano_poc/scripts/make_demo_bundle.py" \
       --out "${HOME}/demo_bundle" \
       --n_boards 50

echo
echo "done. start the demo with:"
echo "  sudo systemctl start aoi-demo"
echo "  ln -sfT ${HOME}/demo_bundle ${WATCH_DIR}/bundle"
echo "  xdg-open http://localhost:8080"
echo
echo "tip: set static IP on Ethernet so the QR code is stable across boots."
