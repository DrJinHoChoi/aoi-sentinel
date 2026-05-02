# Hardware Bill of Materials

## Trainer (on-prem, per customer site)

**NVIDIA DGX Spark** (formerly Project DIGITS, released 2025)

| Spec | Value |
|------|-------|
| Chip | GB10 Grace Blackwell Superchip |
| GPU compute | ~250 TFLOPS FP16 / ~1 PFLOPS FP4 sparse |
| Memory | 128 GB LPDDR5X unified |
| Form factor | 150 × 150 × 50 mm, ~1.2 kg |
| Power | 240 W |
| OS | DGX OS (Ubuntu 22.04, ARM64) |
| CUDA | 12.x |
| Price | ~$3,000 (₩400만 내외) |

**Why this**: 128 GB unified memory lets us train MambaVision-L on full-resolution batches without OOM. ARM64 + NVIDIA's official PyTorch wheels means our entire `mamba-ssm` / `causal-conv1d` stack works natively. Pure on-prem, runs 24/7 from a desk.

### Alternative: x86 if required

**Lenovo ThinkStation P3 Tiny + RTX A2000 12GB** — ~₩300-400만, 1L form factor, factory-integrated GPU, x86 ecosystem. Use this only when a customer's IT requires x86.

## Edge (per line)

**Jetson Orin Nano Developer Kit 8GB** (production)

| Spec | Value |
|------|-------|
| Chip | Ampere GPU + ARM Cortex-A78AE |
| Compute | 67 TOPS sparse INT8 |
| Memory | 8 GB LPDDR5 (shared CPU/GPU) |
| Storage | M.2 NVMe SSD (added separately) |
| Form factor | 100 × 79 × 21 mm, ~250 g (board) |
| Power | 7-15 W |
| OS | JetPack 6 (Ubuntu 22.04, ARM64) |
| Price | ~₩70만 |

**Why this**: enough compute to run a distilled student image encoder (MobileNetV3-Large or ConvNeXt-Pico) at 30+ FPS via TensorRT. Heavy training stays on the trainer; the edge only does inference and operator UI. 24/7 fanless, low-power, mounts on a DIN rail next to the AOI controller.

### POC: existing Jetson Nano 4GB (2019)

| Spec | Value |
|------|-------|
| Chip | Maxwell GPU |
| Compute | ~470 GFLOPS |
| Memory | 4 GB |
| OS | JetPack 4.6 (Ubuntu 18.04, ARM64) |

Sufficient only for the **demo profile** — `LightweightEncoder(size="nano")` (MobileNetV3-Small) at ~5-10 FPS, no training. Useful for showing the "USB plug-and-play" concept to early customers; production deployments must use the Orin Nano.

## Connectivity (edge → trainer / edge → AOI)

- **AOI → edge**: SMB share read-only, or NFS, or vendor SDK callback.
- **Edge → trainer (label sync)**: SMB share read-only on the trainer's side, or SSHFS, or simple `rsync` cron job for the label SQLite file.
- **Trainer → edge (model push)**: shared model registry directory (SMB / NFS).
- All over the customer's existing LAN; no separate networking hardware needed beyond a switch port per box.

## Per-site BOM (for quoting)

For a typical customer with 3 SMT lines:

| Item | Qty | Unit | Total |
|------|-----|------|-------|
| DGX Spark 128 GB | 1 | ₩400만 | ₩400만 |
| Jetson Orin Nano 8 GB Dev Kit | 3 | ₩70만 | ₩210만 |
| NVMe SSD 1TB (per Orin) | 3 | ₩10만 | ₩30만 |
| 7-inch HDMI display + tablet mount | 3 | ₩15만 | ₩45만 |
| Network/power cables, DIN-rail mounts | — | — | ₩20만 |
| **Hardware subtotal** | | | **~₩705만** |

Software/integration is separate and varies by adapter coverage and MES-integration depth.
