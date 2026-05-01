"""Train the 2D ROI false-call classifier.

Run via:
    aoi train classifier --config configs/classifier_2d.yaml
"""
from __future__ import annotations

from pathlib import Path

import yaml


def run(config_path: str | Path) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    # Imported lazily so the base install doesn't need torch.
    import torch
    from torch.utils.data import DataLoader

    from aoi_sentinel.data.dataset import SakiROIDataset
    from aoi_sentinel.models.classifier_2d import build_classifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = SakiROIDataset(cfg["data"]["train_index"], roi_size=cfg["data"]["roi_size"])
    val_ds = SakiROIDataset(cfg["data"]["val_index"], roi_size=cfg["data"]["roi_size"])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
    )

    model = build_classifier(
        backbone=cfg["model"]["backbone"],
        pretrained=cfg["model"]["pretrained"],
        drop_rate=cfg["model"]["drop_rate"],
    ).to(device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        for batch in train_loader:
            x, y, _ = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

        # TODO: validation loop, metrics (precision/recall on TRUE_DEFECT class,
        # false-call reduction rate), checkpoint save, early stop.
        _ = val_loader  # placeholder use
        print(f"epoch {epoch} done")
