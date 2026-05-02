#!/usr/bin/env bash
# Bootstrap an NVIDIA DGX Spark (Grace Blackwell, ARM64, DGX OS) as the
# on-prem trainer for aoi-sentinel.
#
# Run as the unprivileged user (the script will sudo where needed).
#
# Tested on: DGX OS 6 (Ubuntu 22.04 LTS, ARM64), CUDA 12.x.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/aoi-sentinel}"
PY_VER="${PY_VER:-3.11}"

echo "[1/6] system packages"
sudo apt update
sudo apt install -y build-essential git curl python${PY_VER} python${PY_VER}-venv \
                    python3-pip libgl1 libglib2.0-0 sqlite3

echo "[2/6] clone or update repo"
if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone https://github.com/DrJinHoChoi/aoi-sentinel.git "${REPO_DIR}"
fi
cd "${REPO_DIR}"
git pull

echo "[3/6] virtualenv"
python${PY_VER} -m venv .venv
source .venv/bin/activate
pip install -U pip wheel

echo "[4/6] torch + Mamba kernels (ARM64 + CUDA 12.x)"
# DGX Spark ships ARM-native PyTorch wheels via NVIDIA's index.
pip install --extra-index-url https://pypi.nvidia.com torch torchvision
pip install --no-build-isolation "causal-conv1d>=1.5.0"
pip install --no-build-isolation "mamba-ssm>=2.2.4"

echo "[5/6] project install"
pip install -e ".[train,dev,rag,offline_rl]"
pip install timm mambavision

echo "[6/6] sanity check"
python - <<'PY'
import torch, sys
from mamba_ssm import Mamba
import timm
print("python:", sys.version.split()[0])
print("torch :", torch.__version__, "cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
m = timm.create_model("mambavision_tiny_1k", pretrained=False).cuda()
print("MambaVision OK:", m(torch.randn(1, 3, 224, 224).cuda()).shape)
mb = Mamba(d_model=128).cuda()
print("Mamba-SSM OK  :", mb(torch.randn(1, 64, 128).cuda()).shape)
PY

echo "done. trainer can now run: aoi train pretrain --config configs/stage0_pretrain.yaml"
