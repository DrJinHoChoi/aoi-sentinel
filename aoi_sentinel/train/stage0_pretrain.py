"""Stage 0 — supervised pretraining of the image encoder.

Cost-sensitive cross-entropy on benchmark data. Output: image-encoder
weights for the policy's image encoder in stage 1.

Karpathy: look at your data. Stage 0 historically collapsed to the
majority class because (a) the pretrained backbone wasn't getting
ImageNet-normalised input and (b) class weights were a gentle 1:2
which couldn't overcome an 89:11 prior. We fix both: explicit
ImageNet mean/std, plus class weights derived from the actual training
distribution. Print the distribution at start so you don't have to
infer it from a flat val_acc curve.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

# ImageNet stats — every pretrained vision backbone we use was trained with these.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def run(config_path: str | Path) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from aoi_sentinel.data.benchmarks import load_visa
    from aoi_sentinel.models.vmamba import build_image_encoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images, labels, _ = load_visa(cfg["data"]["root"], size=cfg["data"]["roi_size"])
    n = len(images)
    pos_rate = float(labels.mean())
    print(f"[stage0] dataset: {n} images, defect rate = {pos_rate:.4f} "
          f"(majority-class accuracy ceiling = {max(pos_rate, 1 - pos_rate):.4f})")

    # Random shuffle split — board-aware split overkill for benchmark stage.
    idx = np.random.default_rng(cfg["seed"]).permutation(n)
    n_val = int(n * cfg["data"]["val_ratio"])
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    train_pos = float(labels[train_idx].mean())
    print(f"[stage0] train n={len(train_idx)}  val n={len(val_idx)}  train defect rate={train_pos:.4f}")

    train_ds = TensorDataset(
        _to_chw_normalised(images[train_idx]),
        torch.from_numpy(labels[train_idx]).long(),
    )
    val_ds = TensorDataset(
        _to_chw_normalised(images[val_idx]),
        torch.from_numpy(labels[val_idx]).long(),
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["train"]["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=cfg["train"]["num_workers"], pin_memory=True,
    )

    encoder = build_image_encoder(cfg["model"]).to(device)
    head = torch.nn.Linear(encoder.embed_dim, 2).to(device)

    # Class weights from inverse frequency (capped so gradient stays sane).
    # User can override with cfg.train.pos_weight if they want manual control.
    user_override = cfg["train"].get("pos_weight")
    if user_override is None:
        # weight ratio = (1 - pos_rate) / pos_rate, capped at 50:1
        ratio = min((1.0 - train_pos) / max(train_pos, 1e-3), 50.0)
        weights = torch.tensor([1.0, float(ratio)], device=device)
        print(f"[stage0] auto class weights (1:{ratio:.2f}) from train defect rate")
    else:
        weights = torch.tensor([1.0, float(user_override)], device=device)
        print(f"[stage0] using configured class weights (1:{user_override})")

    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    params = list(encoder.parameters()) + list(head.parameters())
    optim = torch.optim.AdamW(params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    out_dir = Path(cfg["train"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg["train"]["epochs"]):
        encoder.train(); head.train()
        train_loss = 0.0; n_train = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = head(encoder(x))
            loss = loss_fn(logits, y)
            optim.zero_grad(); loss.backward(); optim.step()
            train_loss += loss.item() * x.size(0); n_train += x.size(0)
        train_loss /= max(n_train, 1)

        # Validation — track per-class metrics so we catch collapse fast.
        encoder.eval(); head.eval()
        tp = fp = tn = fn = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = head(encoder(x)).argmax(-1)
                tp += int(((pred == 1) & (y == 1)).sum().item())
                fp += int(((pred == 1) & (y == 0)).sum().item())
                tn += int(((pred == 0) & (y == 0)).sum().item())
                fn += int(((pred == 0) & (y == 1)).sum().item())
        total = tp + fp + tn + fn
        acc = (tp + tn) / max(total, 1)
        recall_def = tp / max(tp + fn, 1)        # how many defects we caught
        precision_def = tp / max(tp + fp, 1)     # how many "defect" calls were correct
        n_pred_def = tp + fp
        print(
            f"epoch {epoch}: loss={train_loss:.4f}  acc={acc:.4f}  "
            f"defect_recall={recall_def:.4f}  defect_precision={precision_def:.4f}  "
            f"n_pred_defect={n_pred_def}/{total}"
        )

        torch.save(
            {"encoder": encoder.state_dict(), "head": head.state_dict(),
             "config": cfg, "epoch": epoch},
            out_dir / f"stage0_epoch{epoch:03d}.pt",
        )


def _to_chw_normalised(arr: np.ndarray):
    """uint8 HWC → float CHW with ImageNet normalisation. CRITICAL for
    pretrained backbones — without this the network sees inputs that
    differ from its training distribution by ~3 std and immediately
    collapses to majority-class prediction."""
    import torch
    x = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std
