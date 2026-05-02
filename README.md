# aoi-sentinel

> Vendor-agnostic AOI false-call eliminator. Plug it into any inspection machine — Saki, Koh Young, Mycronic, TRI, Mirtec, Omron — and it learns from operator decisions until false calls converge to zero, with a hard escape-rate constraint throughout.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DrJinHoChoi/aoi-sentinel/blob/main/notebooks/colab_quickstart.ipynb)

## What this does

- **Watches** any AOI machine's output via a thin vendor adapter (~300 LOC per vendor).
- **Decides** per ROI: `DEFECT` / `PASS` / `ESCALATE` (ask the operator).
- **Learns** continuously from operator actions — every rework / scrap / pass click is a label.
- **Self-promotes** safer modes (`SHADOW → ASSIST → AUTONOMOUS`) only when a hold-out replay shows zero escapes and improved false-call rate.
- **Stays on-prem**. No customer data leaves the facility, ever.

## Architecture

```
┌── Customer LAN ─────────────────────────────────────────────────────────────┐
│                                                                              │
│  AOI vendor (Saki / Koh Young / ...)                                         │
│        │ result share                                                        │
│        ▼                                                                     │
│  ┌──────────────────────────────────────────┐                                │
│  │  Edge box  (Jetson Orin Nano 8GB)        │                                │
│  │  - vendor adapter (file watch)           │                                │
│  │  - lightweight image encoder (TRT)       │                                │
│  │  - operator UI (web, mobile-friendly)    │                                │
│  │  - local label queue (SQLite)            │                                │
│  └────────────────┬─────────────────────────┘                                │
│                   │ nightly label sync                                       │
│                   ▼                                                          │
│  ┌──────────────────────────────────────────┐                                │
│  │  Trainer box  (NVIDIA DGX Spark 128GB)   │                                │
│  │  - MambaVision (image)  +  Mamba-SSM     │                                │
│  │    (line history, O(L))                  │                                │
│  │  - Lagrangian PPO under ε-escape         │                                │
│  │  - safety gate (zero escapes on holdout) │                                │
│  │  - atomic model promotion to edges       │                                │
│  └──────────────────────────────────────────┘                                │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

See [docs/deployment_topology.md](docs/deployment_topology.md) for failure modes and networking, [docs/hardware_bom.md](docs/hardware_bom.md) for hardware spec, [docs/vendor_adapter_guide.md](docs/vendor_adapter_guide.md) for adapter contract, and [docs/npi_problem_formulation.md](docs/npi_problem_formulation.md) for the full Constrained-MDP formulation.

## Operating modes

| Mode | Engine role | Operator role | Promotion criterion |
|------|-------------|---------------|---------------------|
| `SHADOW` | shows decision, doesn't act | decides everything (label collection) | 1k+ boards, no escape |
| `ASSIST` | acts when confident, asks when unsure | resolves disagreements | 50k+ boards, FC < 0.5%, 30 days clean |
| `AUTONOMOUS` | decides directly into MES | monitors KPI dashboard | (terminal — auto-demotes on any escape) |

## Layout

```
aoi_sentinel/
├── adapters/                  # vendor plugins (saki, koh_young, generic_csv, ...)
├── runtime/                   # edge daemon, trainer server, label queue,
│                              # model registry, safety gate, mode state machine
├── ui/web/                    # operator card UI (FastAPI + HTMX)
├── models/
│   ├── vmamba/                # MambaVision + Mamba-SSM (trainer side)
│   ├── lightweight/           # MobileNetV3 / ConvNeXt-Pico (edge side)
│   ├── policy/                # Lagrangian PPO actor-critic
│   ├── analyzer_3d/           # Phase 3 (3D height-map analyzer, placeholder)
│   └── rag/                   # Phase 4 (LLM-RAG cause inference, placeholder)
├── sim/                       # NPI streaming env (gymnasium) for offline RL training
├── train/                     # stage0_pretrain + stage1_npi_rl scripts
├── eval/                      # AOI metrics (escape rate, cost curves)
└── data/                      # public benchmark loaders (VisA, DeepPCB, SolDef_AI)

deploy/
├── dgx_spark/                 # trainer setup script (ARM64, CUDA 12)
├── jetson_orin_nano/          # edge setup script + systemd unit
└── trainer_server/            # optional Docker compose

docs/                          # architecture + setup + adapter guide
notebooks/                     # Colab quickstart for Phase 0 pretraining
configs/                       # YAML configs per stage
scripts/                       # benchmark dataset downloaders
```

## Quickstart

### Phase 0 — pretrain the base model (Colab, no on-prem hardware needed yet)

Click the badge above. The notebook clones the repo, installs Mamba kernels, downloads VisA PCB benchmark data, runs Stage 0 (cost-sensitive supervised) and Stage 1 (Lagrangian PPO on the NPI simulator), and plots cost / escape / λ trajectories.

### On-prem deployment

```bash
# Trainer (DGX Spark)
bash deploy/dgx_spark/setup.sh

# Edge (per line, Jetson Orin Nano)
bash deploy/jetson_orin_nano/setup.sh
sudo cp deploy/jetson_orin_nano/aoi-edge.service /etc/systemd/system/
sudo systemctl enable --now aoi-edge
```

Edge auto-discovers the trainer's model registry over the customer LAN; operator UI comes up at `http://<edge-ip>:8080` (web + tablet).

## Why this matters

- **Vendor-agnostic** — one engine across every AOI in the factory; no vendor lock-in.
- **Plug-and-play** — adding a new line = drop in a Jetson Orin Nano + edit one config line.
- **On-prem only** — no SaaS, no customer-data egress; competitive for automotive QA.
- **Self-improving** — false-call rate falls as boards accumulate; operator workload drops accordingly.
- **Hard safety guarantee** — escape rate is gated, not optimised. A miss demotes the system automatically.

## Open standard

The data schema and adapter contract this engine consumes are defined as a vendor-neutral open standard:

> **[aoi-common-spec](https://github.com/DrJinHoChoi/aoi-common-spec)** — RFC v0.1, BSD 3-Clause.

`aoi-sentinel` is the reference implementation. Anyone — including AOI vendors — can build conforming adapters or readers without licensing this engine. This separation is intentional: we believe the inspection schema should be a public commons, with implementations competing on top.

## Why Mamba

No peer-reviewed Mamba-on-SMT/PCB work exists as of 2026-04. The closest neighbour is [MambaAD](https://arxiv.org/abs/2404.06564) (NeurIPS 2024) on MVTec/VisA. We use Mamba twice — as the image encoder ([MambaVision](https://arxiv.org/abs/2407.08083), CVPR 2025) and as the linear-time line-history sequence encoder. Combining that with online constrained RL for cost-asymmetric AOI is, to our knowledge, an open niche. See [docs/sota_landscape.md](docs/sota_landscape.md).

## Strategy

We treat strategy as code: see [docs/strategic_brief.md](docs/strategic_brief.md) for the durable framing of company decisions (Gates / Buffett / Musk / Karpathy synthesis), and [docs/sales/](docs/sales/) for the live pilot offer and outreach templates.

## License

Proprietary — internal use only. The open standard at [aoi-common-spec](https://github.com/DrJinHoChoi/aoi-common-spec) is BSD 3-Clause.
