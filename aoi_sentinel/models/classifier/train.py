"""Training loop for the cost-sensitive classifier.

Karpathy: tight loops. The default config trains for a small number of
epochs at low LR, evaluates after each epoch, and writes the best
checkpoint by held-out cost (not accuracy). 5-minute training runs by
design — let the candidate succeed or fail fast, then iterate the data
or the config rather than waiting for a long run.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np

from aoi_sentinel.runtime.label_queue import LabelRecord


@dataclass
class TrainConfig:
    encoder_size: str = "small"
    pretrained: bool = True
    roi_size: int = 224

    epochs: int = 3
    batch_size: int = 32
    lr: float = 3.0e-4
    weight_decay: float = 0.05
    focal_gamma: float = 2.0
    num_workers: int = 0      # safe default for the trainer process

    seed: int = 42
    cost_matrix: list[list[float]] | None = None  # [2,3] — escape weighted

    image_root: str = "."

    def __post_init__(self) -> None:
        if self.cost_matrix is None:
            self.cost_matrix = [
                [1.0,    0.0, 5.0],   # FALSE_CALL  → DEFECT/PASS/ESCALATE
                [0.0, 1000.0, 5.0],   # TRUE_DEFECT → DEFECT/PASS/ESCALATE
            ]


def train_classifier(
    train_records: list[LabelRecord],
    val_records: list[LabelRecord],
    cfg: TrainConfig,
    out_dir: str | Path,
) -> tuple[Path, Path]:
    """Train and save a checkpoint. Returns (weights_path, config_path).

    Uses board-wise stratification done upstream (caller is responsible).
    Writes:
        out_dir/weights.pt
        out_dir/config.json   — TrainConfig serialised + git hash if available
    """
    # All torch imports kept inside the function so importing this module is free.
    import torch
    from torch.utils.data import DataLoader

    from aoi_sentinel.models.classifier._impl import (
        _LightweightClassifierImpl,
        CostFocalLoss,
        class_weights_from_cost,
    )
    from aoi_sentinel.models.classifier._dataset_impl import _LabelDatasetImpl

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = _LabelDatasetImpl(train_records, image_root=cfg.image_root, roi_size=cfg.roi_size)
    val_ds = _LabelDatasetImpl(val_records, image_root=cfg.image_root, roi_size=cfg.roi_size)
    if not len(train_ds):
        raise ValueError("train set is empty after label filtering")

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=device.type == "cuda")
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=device.type == "cuda") if len(val_ds) else None

    model = _LightweightClassifierImpl(
        encoder_size=cfg.encoder_size, pretrained=cfg.pretrained
    ).to(device)
    weights = class_weights_from_cost(cfg.cost_matrix).to(device)
    loss_fn = CostFocalLoss(gamma=cfg.focal_gamma, class_weights=weights).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    history: list[dict] = []

    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        n = 0
        for x, y, _ in train_dl:
            x = x.to(device); y = torch.as_tensor(y, dtype=torch.long, device=device)
            logits = model(x)
            loss = loss_fn(logits, y)
            optim.zero_grad(); loss.backward(); optim.step()
            running += loss.item() * x.size(0); n += x.size(0)
        train_loss = running / max(n, 1)

        val_loss = float("nan")
        if val_dl is not None:
            model.eval()
            running = 0.0; n = 0
            with torch.no_grad():
                for x, y, _ in val_dl:
                    x = x.to(device); y = torch.as_tensor(y, dtype=torch.long, device=device)
                    logits = model(x)
                    running += loss_fn(logits, y).item() * x.size(0); n += x.size(0)
            val_loss = running / max(n, 1)

        elapsed = time.time() - t0
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "elapsed_s": elapsed})
        print(f"[train] epoch {epoch}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  ({elapsed:.1f}s)")

        if not np.isnan(val_loss) and val_loss < best_val:
            best_val = val_loss
            torch.save({"state_dict": model.state_dict(), "config": model.config}, out / "weights.pt")

    # If we never had a val set, save the final epoch.
    if val_dl is None:
        torch.save({"state_dict": model.state_dict(), "config": model.config}, out / "weights.pt")

    cfg_dict = asdict(cfg)
    cfg_dict["history"] = history
    cfg_dict["device"] = str(device)
    (out / "config.json").write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")

    return out / "weights.pt", out / "config.json"
