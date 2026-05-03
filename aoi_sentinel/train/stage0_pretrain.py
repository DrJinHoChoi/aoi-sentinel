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
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

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

    train_labels = labels[train_idx]
    train_pos = float(train_labels.mean())
    print(f"[stage0] train n={len(train_idx)}  val n={len(val_idx)}  train defect rate={train_pos:.4f}")

    train_ds = TensorDataset(
        _to_chw_normalised(images[train_idx]),
        torch.from_numpy(train_labels).long(),
    )
    val_ds = TensorDataset(
        _to_chw_normalised(images[val_idx]),
        torch.from_numpy(labels[val_idx]).long(),
    )

    # Balanced batch sampler — every batch is ~50/50 class. Heavy class
    # weights alone weren't enough on a 91:9 split; without sampler the
    # gradient signal from the rare class is too noisy and the model
    # collapses to "always predict majority". WeightedRandomSampler fixes
    # this at the data layer rather than fighting it in the loss layer.
    sample_weights = np.where(train_labels == 1, 1.0 / max(train_pos, 1e-3),
                                                  1.0 / max(1.0 - train_pos, 1e-3))
    sampler = WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(train_labels),  # one epoch = same-size as the original train set
        replacement=True,
    )
    print(f"[stage0] using WeightedRandomSampler — each batch ≈ 50/50 class balance")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"],
        sampler=sampler,                                # mutually exclusive with shuffle=True
        num_workers=cfg["train"]["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=cfg["train"]["num_workers"], pin_memory=True,
    )

    encoder = build_image_encoder(cfg["model"]).to(device)
    head = torch.nn.Linear(encoder.embed_dim, 2).to(device)

    # With WeightedRandomSampler the model trains on a balanced distribution,
    # so DON'T pre-bias the head toward the natural prior — that would fight
    # the sampler and cause overshoot to the other side. Initialise at
    # neutral [0, 0] and let the sampled gradient settle the equilibrium.
    with torch.no_grad():
        head.bias.zero_()
    print(f"[stage0] head bias initialised at neutral [0, 0] (paired with balanced sampler)")

    # Likewise drop class weights to 1:1 — the sampler already supplies
    # balanced batches, so any extra weighting just causes overcorrection
    # (we previously saw the model flip from "always 0" to "always 1" with
    # weight 1:2). Honour an explicit override from the config if set.
    user_override = cfg["train"].get("pos_weight")
    pos_weight = float(user_override) if user_override is not None else 1.0
    weights = torch.tensor([1.0, pos_weight], device=device)
    print(f"[stage0] class weights (1:{pos_weight}) — neutral; sampler handles imbalance")

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
        # We compute two threshold settings:
        #   default:    argmax (0.5 prob threshold) — calibrated for the
        #               sampler's 50/50 training distribution
        #   prior-adj:  shift the decision boundary by log(prior) so the
        #               model behaves correctly on the natural 91/9 val set
        encoder.eval(); head.eval()
        tp = fp = tn = fn = 0
        tp_p = fp_p = tn_p = fn_p = 0
        prior_logit_shift = float(np.log(max(1.0 - train_pos, 1e-3) / max(train_pos, 1e-3)))
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = head(encoder(x))
                # Default argmax
                pred = logits.argmax(-1)
                tp += int(((pred == 1) & (y == 1)).sum().item())
                fp += int(((pred == 1) & (y == 0)).sum().item())
                tn += int(((pred == 0) & (y == 0)).sum().item())
                fn += int(((pred == 0) & (y == 1)).sum().item())
                # Prior-shifted: predict class 1 only if logit_1 - logit_0 > log(p0/p1)
                margin = logits[:, 1] - logits[:, 0]
                pred_p = (margin > prior_logit_shift).long()
                tp_p += int(((pred_p == 1) & (y == 1)).sum().item())
                fp_p += int(((pred_p == 1) & (y == 0)).sum().item())
                tn_p += int(((pred_p == 0) & (y == 0)).sum().item())
                fn_p += int(((pred_p == 0) & (y == 1)).sum().item())
        total = tp + fp + tn + fn
        acc = (tp + tn) / max(total, 1)
        recall_def = tp / max(tp + fn, 1)
        precision_def = tp / max(tp + fp, 1)
        n_pred_def = tp + fp
        acc_p = (tp_p + tn_p) / max(total, 1)
        recall_p = tp_p / max(tp_p + fn_p, 1)
        precision_p = tp_p / max(tp_p + fp_p, 1)
        n_pred_p = tp_p + fp_p
        print(
            f"epoch {epoch}: loss={train_loss:.4f}  "
            f"argmax → acc={acc:.3f} rec={recall_def:.3f} prec={precision_def:.3f} npred={n_pred_def}  | "
            f"prior-adj → acc={acc_p:.3f} rec={recall_p:.3f} prec={precision_p:.3f} npred={n_pred_p}"
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
