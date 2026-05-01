# Dev Environment — WSL2 Ubuntu 22.04

`mamba-ssm` and `causal-conv1d` ship custom CUDA kernels that **do not build natively on Windows**. Use WSL2 for development; deploy to Linux containers in production.

## 1. WSL2 prerequisites

```powershell
# In an admin PowerShell on Windows:
wsl --install -d Ubuntu-22.04
wsl --set-default-version 2
```

NVIDIA driver on Windows host already exposes the GPU to WSL2 — no driver inside WSL.

## 2. Inside WSL Ubuntu 22.04

```bash
sudo apt update && sudo apt install -y build-essential git python3.10 python3.10-venv python3-pip

# CUDA toolkit 12.1 (matches torch 2.4 wheels)
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run --silent --toolkit
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
nvidia-smi  # should show your GPU
```

## 3. Project setup

```bash
cd ~ && git clone git@github.com:DrJinHoChoi/aoi-sentinel.git
cd aoi-sentinel
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel

# Torch first (must match CUDA 12.1)
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Mamba kernels
pip install causal-conv1d>=1.4.0
pip install mamba-ssm>=2.2.2

# Project
pip install -e ".[train,rag,dev]"
```

## 4. Sanity check

```bash
python - <<'PY'
import torch
from mamba_ssm import Mamba
import timm
m = timm.create_model('mambavision_tiny_1k', pretrained=False).cuda()
x = torch.randn(2, 3, 224, 224).cuda()
print('vision out:', m(x).shape)
mb = Mamba(d_model=128).cuda()
y = torch.randn(2, 64, 128).cuda()
print('seq out:', mb(y).shape)
PY
```

If both print without error you are ready.

## 5. Pinned versions (verified working)

```
python==3.10.*
cuda==12.1
torch==2.4.0
torchvision==0.19.0
causal-conv1d==1.4.0
mamba-ssm==2.2.2
timm==1.0.11
mambavision==1.0.*  # pip install mambavision
```

Drift from these and the kernels may fail to load.

## 6. Native Windows (not recommended)

Only if WSL2 is impossible:

```powershell
# Use community-built wheels — DO NOT compile from source on Windows.
pip install <prebuilt-causal-conv1d-windows.whl>
pip install <prebuilt-mamba-ssm-windows.whl>
```

Maintained ports lag upstream; only viable for read-only inference, not training.
