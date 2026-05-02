"""Empirical Mamba vs Transformer benchmark on the inspection-history sequence.

Karpathy: every "X is faster" claim must come with a number. This script
forwards a fixed batch through both encoders at increasing sequence
lengths and prints the latency + memory table — same parameter count,
same input shape, only the temporal mixer differs.

Run on a Colab T4 / L4 / A100:

    python scripts/bench_mamba_vs_transformer.py --max-len 4096 --batch 4

Expected behaviour (qualitative):
  - Mamba latency scales roughly linearly with L
  - Transformer latency scales quadratically with L
  - At L=4096 Mamba should be ~4-10× faster on a T4
  - Mamba peak memory grows ~2× with L; transformer ~4× (the attention
    matrix dominates)

Output: stdout table + optional --json path.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max-len", type=int, default=4096)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeat", type=int, default=10)
    p.add_argument("--json", type=str, default=None)
    args = p.parse_args()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  d_model={args.d_model}  layers={args.layers}  batch={args.batch}")

    # Build encoders once; reuse across L.
    from aoi_sentinel.models.policy._transformer_impl import _TransformerSeqEncImpl
    txfm = _TransformerSeqEncImpl(
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=4,
        max_seq_len=args.max_len,
    ).to(device).eval()

    try:
        from aoi_sentinel.models.vmamba.sequence_encoder import SequenceEncoder
        mamba = SequenceEncoder(
            d_model=args.d_model,
            n_layers=args.layers,
        ).to(device).eval()
    except Exception as e:
        print(f"WARN: mamba unavailable ({e}); will only bench transformer")
        mamba = None

    rows = []
    seq_lens = [L for L in (256, 512, 1024, 2048, 4096) if L <= args.max_len]

    for L in seq_lens:
        history = _make_history(args.batch, L, device)
        row = {"L": L}
        for name, enc in [("transformer", txfm), ("mamba", mamba)]:
            if enc is None:
                row[f"{name}_ms"] = None; row[f"{name}_mem_mb"] = None
                continue
            ms, mem_mb = _bench(enc, history, args.warmup, args.repeat, device)
            row[f"{name}_ms"] = ms
            row[f"{name}_mem_mb"] = mem_mb
        if row["transformer_ms"] and row.get("mamba_ms"):
            row["speedup"] = row["transformer_ms"] / row["mamba_ms"]
        rows.append(row)

    _print_table(rows)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_history(b: int, L: int, device):
    import torch
    h = torch.zeros((b, L, 5), dtype=torch.float32, device=device)
    h[..., 0] = torch.randint(0, 1000, (b, L), device=device).float()  # image_idx
    h[..., 1] = torch.randint(0, 2, (b, L), device=device).float()    # saki call
    h[..., 2] = torch.randint(0, 4, (b, L), device=device).float()    # action (incl no-action)
    h[..., 3] = torch.randint(0, 2, (b, L), device=device).float()    # mask
    h[..., 4] = torch.randint(0, 2, (b, L), device=device).float()    # label
    return h


def _bench(enc, history, warmup: int, repeat: int, device):
    import torch
    torch.cuda.empty_cache() if device.type == "cuda" else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for _ in range(warmup):
            enc(history)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeat):
            enc(history)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / repeat
    mem_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if device.type == "cuda"
        else float("nan")
    )
    return elapsed * 1000.0, mem_mb


def _print_table(rows):
    print()
    print(f"{'L':>6}  {'transformer (ms)':>18}  {'mamba (ms)':>14}  {'speedup':>8}  {'txfm mem (MB)':>14}  {'mamba mem (MB)':>14}")
    print("-" * 90)
    for r in rows:
        sp = r.get("speedup")
        sp_s = f"{sp:7.2f}x" if sp else "    n/a"
        print(
            f"{r['L']:>6}  "
            f"{(r['transformer_ms'] if r['transformer_ms'] else float('nan')):>18.2f}  "
            f"{(r['mamba_ms'] if r['mamba_ms'] else float('nan')):>14.2f}  "
            f"{sp_s:>8}  "
            f"{(r['transformer_mem_mb'] if r['transformer_mem_mb'] else float('nan')):>14.1f}  "
            f"{(r['mamba_mem_mb'] if r['mamba_mem_mb'] else float('nan')):>14.1f}"
        )


if __name__ == "__main__":
    main()
