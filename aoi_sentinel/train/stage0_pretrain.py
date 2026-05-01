"""Stage 0: supervised pretraining of the MambaVision image encoder.

Cost-sensitive cross-entropy on benchmark data. The result is the cold-start
weights for the policy's image encoder in stage 1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


def run(config_path: str | Path) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from aoi_sentinel.data.benchmarks import load_visa
    from aoi_sentinel.models.vmamba import build_image_encoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images, labels, _ = load_visa(cfg["data"]["root"], size=cfg["data"]["roi_size"])

    # Stratified board-aware split is overkill for the benchmark stage; use a simple shuffle.
    idx = np.random.default_rng(cfg["seed"]).permutation(len(images))
    n_val = int(len(idx) * cfg["data"]["val_ratio"])
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    def _to_chw(arr):
        return torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0

    train_ds = TensorDataset(_to_chw(images[train_idx]), torch.from_numpy(labels[train_idx]))
    val_ds = TensorDataset(_to_chw(images[val_idx]), torch.from_numpy(labels[val_idx]))
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["train"]["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["num_workers"], pin_memory=True)

    encoder = build_image_encoder(cfg["model"]).to(device)
    head = torch.nn.Linear(encoder.embed_dim, 2).to(device)

    pos_weight = float(cfg["train"].get("pos_weight", 1.0))
    weights = torch.tensor([1.0, pos_weight], device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    params = list(encoder.parameters()) + list(head.parameters())
    optim = torch.optim.AdamW(params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    out_dir = Path(cfg["train"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg["train"]["epochs"]):
        encoder.train()
        head.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = head(encoder(x))
            loss = loss_fn(logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

        encoder.eval()
        head.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = head(encoder(x)).argmax(-1)
                correct += int((pred == y).sum().item())
                total += int(y.numel())
        acc = correct / max(total, 1)
        print(f"epoch {epoch}: val_acc={acc:.4f}")

        torch.save(
            {"encoder": encoder.state_dict(), "head": head.state_dict()},
            out_dir / f"stage0_epoch{epoch:03d}.pt",
        )
