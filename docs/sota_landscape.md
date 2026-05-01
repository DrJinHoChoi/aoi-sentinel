# SOTA Landscape — SMT AOI False-Call AI

Compiled 2026-04 from background-research agents (web + literature). Verify links before committing engineering effort.

## 1. Industry / commercial state

No public benchmark exists for SMT AOI false-call reduction specifically. The market is owned by proprietary vendor solutions:

- **[Mycronic DeepReview (2024)](https://smttoday.com/2024/04/04/mycronic-promises-to-eliminate-majority-of-false-calls-with-new-deep-learning-system-for-3d-aoi/)** — 50–90% false-call removal claimed.
- **Koh Young KSMART** — Korean, dominant in 3D AOI/SPI.
- **[Siemens Opcenter AOI-FCR](https://plm.sw.siemens.com/en-US/opcenter/manufacturing-intelligence/aoi-fcr/)**.
- **TRI / Cyberoptics / Mirtec / Omron / Saki** — internal AI modules.

None publish datasets, model architectures, or detailed metrics. Result: comparing "SOTA" requires building our own benchmark.

## 2. Adjacent academic SOTA

| Domain | Top method (2024–2026) | Notes |
|--------|------------------------|-------|
| Anomaly detection (MVTec / VisA) | [**MambaAD**](https://arxiv.org/abs/2404.06564) (NeurIPS 2024), EfficientAD | Mamba already SOTA |
| PCB defect detection | [SDD-Net](https://www.sciencedirect.com/science/article/abs/pii/S0925231224013468) (YOLOv7-tiny variant), 99.1% mAP | Bare PCB, not SMT components |
| Vision-Mamba classification | [Spatial-Mamba](https://arxiv.org/abs/2410.15091) (ICLR 2025, 84.6%), [MambaVision-L](https://arxiv.org/abs/2407.08083) (CVPR 2025, 85.0%) | MambaVision is most production-ready |
| Solder joint inspection | [Springer 2025 — front+lateral fusion](https://link.springer.com/article/10.1007/s00170-025-15460-8) | Real production line deployment |
| Cautionary | [MambaOut](https://arxiv.org/abs/2405.07992) (CVPR 2025) | Mamba may be unnecessary for plain 224² classification |

**Open niche**: peer-reviewed Mamba-on-PCB/SMT — only 2–3 arXiv preprints. Genuine research opportunity.

## 3. Backbone choice — MambaVision-T/S

Picked over plain VMamba because:
- NVIDIA-maintained, timm-registered (`create_model('mambavision_tiny_1k', pretrained=True)`)
- Hybrid Mamba + self-attention in late stages — transfers more cleanly under distribution shift
- HuggingFace-hosted weights (`nvidia/MambaVision-*-1K`)
- ONNX export is workable (hybrid attention path)

Always run a **ConvNeXt-T baseline** alongside — if Mamba is within 0.5% accuracy, ship the simpler model (MambaOut warning).

## 4. Sequence encoder — Mamba-SSM (vanilla)

For inspection-history sequence encoding (`L=512–4096`), use `mamba-ssm` directly. Linear-time selectivity beats transformer self-attention at our sequence lengths.

## 5. Algorithm choice for online learning under escape constraint

For the NPI online learning setting, the relevant literature is:
- **Constrained MDPs** — Altman 1999
- **[Lagrangian PPO / CPO](https://arxiv.org/abs/1705.10528)** — Achiam et al., constrained policy optimization
- **[Conservative Safety Critics](https://arxiv.org/abs/2010.14497)** — Bharadhwaj et al., safety critic in actor-critic loop
- **[Decision Mamba](https://arxiv.org/abs/2403.19925)** — sequence modeling RL with Mamba (relevant if/when we move to multi-step horizons)
- **[Selective Classification](https://arxiv.org/abs/1705.08500)** — Geifman & El-Yaniv, used as the warm-start policy for safe exploration

## 6. Library stack

| Concern | Pick | Reason |
|---------|------|--------|
| Image backbone | `mambavision`, `timm` | Pretrained, easy API |
| Sequence encoder | `mamba-ssm`, `causal-conv1d` | Reference SSM kernel |
| Env interface | `gymnasium` | Standard |
| RL trainer | Custom Lagrangian PPO (~400 lines) | No mature constrained-online RL lib; `omnisafe` is closest but heavyweight |
| Offline RL fallback | `d3rlpy` | If we revisit batch-mode fine-tuning |
| Logging | `wandb` or `tensorboard` | Either |

## 7. Datasets (Phase 0 pretraining)

| Dataset | Use | License | Link |
|---------|-----|---------|------|
| **VisA** (PCB1–4) | Anomaly-style pretrain | CC BY 4.0 (commercial OK) | [GitHub](https://github.com/amazon-science/spot-diff) |
| **DeepPCB** | Trace defects + template/test pairs | Academic | [GitHub](https://github.com/tangsanli5201/DeepPCB) |
| **DsPCBSD+** (2024) | Larger PCB defect set | CC BY 4.0 | [GitHub](https://github.com/Powerteach/DsPCBSD-) |
| **SolDef_AI** | Solder-joint level — direct match | CC BY 4.0 | [Mendeley](https://www.sciencedirect.com/science/article/pii/S2352340924005225) |
| **AI-Hub PCB / SMT 외관** | Korean SMT — best domain match | AI-Hub registration | [aihub.or.kr](https://www.aihub.or.kr) |
| **MVTec AD / LOCO** | Texture / logical anomalies — research only | CC BY-NC-SA | [MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad) |

**License gotcha**: MVTec is non-commercial. Production weights must be derived only from VisA, DsPCBSD+, SolDef_AI, AI-Hub.

## 8. Honest gaps

1. No peer-reviewed Mamba-on-SMT benchmark — we are creating one.
2. ONNX export for pure SS2D blocks is rough; MambaVision exports cleanly because of attention path.
3. Online constrained RL libraries are immature; expect to write our own trainer.
4. `mamba-ssm` does not build natively on Windows — develop in WSL2.
