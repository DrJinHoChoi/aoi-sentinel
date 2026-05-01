# aoi-sentinel

> Mamba-RL false-call filter for SMT AOI (Saki) — NPI online learning under hard escape constraint.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DrJinHoChoi/aoi-sentinel/blob/main/notebooks/colab_quickstart.ipynb)

## What this does

Saki AOI flags ~30% **false calls** (NG that's actually OK) on automotive electronics SMT lines. When a **new product** enters production (NPI), there is no product-specific labeled data on day 0 — and Saki's rule-based call rate stays high until manually retuned.

aoi-sentinel learns the acceptance policy online from operator decisions, **gated by a hard escape-rate constraint** (a missed true defect is ~1000× more costly than a false call). The policy improves as labels accumulate, while the constraint guarantees no escape regression during exploration.

## Architecture

```
[Saki ROI image]                    [Inspection history (last L)]
        │                                       │
        ▼                                       ▼
 MambaVision (image)            Mamba-SSM (sequence, O(L))
        │                                       │
        └──────────────────┬────────────────────┘
                           ▼
              Actor-critic with two value heads
              ├── action ∈ {DEFECT, PASS, ESCALATE}
              ├── V (reward)
              └── V_c (escape cost)

           Lagrangian PPO update with dual-ascent on λ
```

See [docs/npi_problem_formulation.md](docs/npi_problem_formulation.md) for the full Constrained-MDP formulation.

## Phases

| Phase | Goal | Data | Notes |
|-------|------|------|-------|
| 0 | Pretrain MambaVision | Public PCB/SMT benchmarks | Cost-sensitive supervised |
| 1 | Build NPI simulator | Same datasets, replayed | Gymnasium env + cost matrix |
| 2 | Mamba RL agent | NPI simulator | Lagrangian PPO under ε-escape |
| 3 | Real Saki line | Live stream | Cold-start with Phase 0 weights |
| 4 | 3D analyzer + LLM-RAG | Saki 3D + history DB | Separate modules |

## Layout

```
aoi_sentinel/
├── data/
│   ├── saki.py                # Real Saki dump parser (placeholder, fill on first sample)
│   ├── dataset.py             # ROI Dataset for supervised stages
│   └── benchmarks/            # VisA / DeepPCB / SolDef_AI loaders
├── sim/                       # NPI streaming env (gymnasium)
├── models/
│   ├── vmamba/                # MambaVision image encoder + Mamba-SSM sequence encoder
│   ├── policy/                # Actor-critic + Lagrangian PPO
│   ├── analyzer_3d/           # Phase 3 (placeholder)
│   └── rag/                   # Phase 4 (placeholder)
├── train/
│   ├── stage0_pretrain.py     # Supervised on benchmarks
│   └── stage1_npi_rl.py       # Online RL on simulator
├── eval/                      # AOI metrics (escape rate, cost curves)
└── serve/                     # FastAPI inference

configs/                       # YAML configs per stage
docs/                          # Architecture + setup docs
scripts/                       # Benchmark dataset downloaders
```

## Quickstart

**Easiest — Google Colab** (T4 free, A100 on Pro): click the badge above or follow [docs/setup_colab.md](docs/setup_colab.md). The notebook clones, installs, downloads VisA, and runs both stages end-to-end.

**Local — WSL2 Ubuntu** (required for Mamba CUDA kernels): see [docs/setup_wsl.md](docs/setup_wsl.md).

```bash
# 1. install
pip install -e ".[train,dev]"

# 2. fetch benchmark data
python scripts/download_visa.py --out data/raw/visa --pcb-only
python scripts/download_deeppcb.py --out data/raw/deeppcb

# 3. Phase 0 — pretrain image encoder
aoi train pretrain --config configs/stage0_pretrain.yaml

# 4. Phase 1 — Mamba RL on NPI simulator
aoi train npi-rl --config configs/stage1_npi_rl.yaml
```

## Why this is novel

No published Mamba-on-SMT/PCB peer-reviewed work exists as of 2026-04. The closest neighbor is **MambaAD** (NeurIPS 2024) on MVTec/VisA. Combining Mamba (image + sequence) with **online constrained RL for cost-asymmetric AOI** is, to our knowledge, an open niche.

See [docs/sota_landscape.md](docs/sota_landscape.md) for the full landscape review.

## License

Proprietary — internal use only.
