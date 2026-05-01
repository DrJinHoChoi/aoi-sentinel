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

```python
!pip install -q packaging ninja
!pip install -q causal-conv1d==1.4.0 --no-build-isolation
!pip install -q mamba-ssm==2.2.2 --no-build-isolation
```

The `--no-build-isolation` flag is critical — without it pip starts a fresh environment that doesn't see Colab's pre-installed torch, and the CUDA build fails.

First install compiles for ~5–10 min. The notebook caches it for subsequent sessions.

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
