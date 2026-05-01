# Colab Setup

The fastest path to a first run — no local Linux setup, GPU included.

[**Open in Colab**](https://colab.research.google.com/github/DrJinHoChoi/aoi-sentinel/blob/main/notebooks/colab_quickstart.ipynb)

## Tier guide

| Tier | GPU | VRAM | Recommended config | Stage 0 time | Stage 1 time |
|------|-----|------|--------------------|--------------|--------------|
| Free | T4 | 16 GB | `*_colab.yaml` | ~30–60 min | ~30–60 min |
| Pro | L4 / A100 | 24 / 40 GB | full `stage0_pretrain.yaml` / `stage1_npi_rl.yaml` | ~15 min | ~30 min |
| Pro+ | A100 | 40 GB | full + larger MambaVision-S | ~25 min | ~45 min |

T4 config trims:
- batch 32 (vs 64), 8 epochs (vs 30)
- 300 RL iterations × 256 rollout steps (vs 2000 × 1024)
- sequence d_model 192 (vs 256), n_layers 3 (vs 4)
- frozen image encoder during RL (saves activations memory)

## Mamba install on Colab

Colab's torch version drifts every few weeks; pinned wheels go stale fast. The notebook uses a 3-tier strategy in cells §2:

1. **Tier 1** — pip-resolved latest with `--no-build-isolation` (so it reuses Colab's torch):
   ```bash
   pip install --no-build-isolation "causal-conv1d>=1.5.0"
   pip install --no-build-isolation "mamba-ssm>=2.2.4"
   ```
2. **Tier 2** — explicit wheel URL constructed from the detected `(torch, python, cuda, abi)` tuple. Wheels live on each project's GitHub releases page:
   - https://github.com/Dao-AILab/causal-conv1d/releases
   - https://github.com/state-spaces/mamba/releases
3. **Tier 3** — pure-PyTorch path (no custom CUDA kernels). Slower but always works:
   ```bash
   MAMBA_SKIP_CUDA_BUILD=TRUE CAUSAL_CONV1D_SKIP_CUDA_BUILD=TRUE \
     pip install --no-build-isolation causal-conv1d mamba-ssm
   ```

`--no-build-isolation` is critical — without it pip spins up a fresh env that doesn't see Colab's pre-installed torch, and the CUDA build fails.

If Tier 1 + Tier 2 both fail, browse the releases pages, copy the wheel URL closest to your torch version, and paste it directly. The wheel filename pattern is `<pkg>-<ver>+cu<MAJ>torch<MM>cxx11abi(TRUE|FALSE)-<py>-<py>-linux_x86_64.whl`.

## Persistence

Free Colab disconnects after ~12h idle / 24h max. To keep checkpoints and data across sessions, mount Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then point `out_dir` and `data.root` in the configs at `/content/drive/MyDrive/aoi-sentinel/...`.

## Common issues

- **"causal-conv1d build failed"**: Colab updated its torch and the wheel is stale. Pin a different `causal-conv1d` minor (1.4.0 → 1.5.0) or fall back to a torch version known to work — see [docs/setup_wsl.md](setup_wsl.md) pin matrix.
- **OOM on T4**: drop `batch_size` to 16 in `stage0_pretrain_colab.yaml`. If still OOM, use `model.size: tiny` (already default) and `freeze_backbone: true`.
- **Disconnects mid-train**: enable Drive mount + checkpoint every epoch (already the default in stage 0). Stage 1 saves every 25 iters.
