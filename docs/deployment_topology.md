# Deployment Topology

```
┌─── Customer site (single facility) ─────────────────────────────────────────┐
│                                                                              │
│   Line 1                Line 2               Line N                          │
│   ┌────────┐            ┌────────┐           ┌────────┐                      │
│   │ AOI    │            │ AOI    │   ...     │ AOI    │   (Saki / Koh Young  │
│   │ vendor │            │ vendor │           │ vendor │    / Mycronic / ...) │
│   └───┬────┘            └───┬────┘           └───┬────┘                      │
│       │ result share         │                    │                          │
│       ▼                       ▼                    ▼                          │
│   ┌─────────┐             ┌─────────┐          ┌─────────┐                   │
│   │ Edge box│             │ Edge box│          │ Edge box│                   │
│   │ (Orin   │             │ (Orin   │          │ (Orin   │                   │
│   │  Nano)  │             │  Nano)  │          │  Nano)  │                   │
│   │ adapter │             │ adapter │          │ adapter │                   │
│   │ + UI    │             │ + UI    │          │ + UI    │                   │
│   │ + label │             │ + label │          │ + label │                   │
│   │ queue   │             │ queue   │          │ queue   │                   │
│   └────┬────┘             └────┬────┘          └────┬────┘                   │
│        │                       │                     │                         │
│        └───────────────┬───────┴─────────────────────┘                         │
│                        ▼                                                       │
│                 ┌──────────────┐                                              │
│                 │ Trainer box  │                                              │
│                 │ (DGX Spark)  │   pulls labels nightly,                      │
│                 │ - registry   │   trains, gates, promotes                    │
│                 │ - safety gate│                                              │
│                 │ - retraining │                                              │
│                 └──────────────┘                                              │
│                        │                                                       │
│                        └─►  registry shared (NFS / SMB) back to all edges     │
│                                                                                │
│  All boxes inside customer LAN. No data leaves the facility.                  │
└────────────────────────────────────────────────────────────────────────────────┘
```

## Hardware

| Role                  | Recommended                              | Quantity        | Approx. price |
|-----------------------|-------------------------------------------|-----------------|---------------|
| Trainer (per facility)| **NVIDIA DGX Spark 128GB**                | 1 per site      | ~₩400만        |
| Edge (per line)       | **Jetson Orin Nano 8GB Dev Kit**          | 1 per line      | ~₩70만         |
| POC demo edge         | Jetson Nano 4GB (existing)                | 1 (you have it) | —              |
| Trainer (alt — x86)   | Lenovo ThinkStation P3 Tiny + RTX A2000   | 1 per site      | ~₩400만        |

## Networking

- All on customer LAN.
- Edge boxes mount the AOI vendor's result share (SMB / NFS) read-only.
- Trainer mounts each edge's `label_queue.db` (SMB / SSHFS) read-only and writes the model registry to a share that edges mount read-only.
- No outbound internet required at runtime. Updates are delivered via the trainer (which can pull from GitHub on a maintenance window).

## Failure modes & graceful degradation

| What fails | What still works |
|------------|------------------|
| Trainer down | Edge keeps inferring with current model; labels keep accumulating; model just doesn't improve until trainer is back |
| One edge down | That line falls back to vendor-only inspection (no harm) |
| Adapter crash | Edge restarts adapter; engine pauses for that line until source is reachable |
| Model promotion produces a bad model | Safety gate rejects before promotion; if a bad model slips through, an escape triggers automatic demotion to ASSIST mode |

## Security & data governance

- Customer data **never leaves the facility**.
- Models trained on customer data are stored on the trainer; we do not ship them between customers.
- The pretrained Phase-0 base model (trained only on public benchmarks: VisA, DeepPCB, AI-Hub) is the only artifact we deliver.
