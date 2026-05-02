"""Torch implementation kept here so importing the module above doesn't
require torch in environments that only need the schema (Windows, CI lint)."""
from __future__ import annotations

import torch
import torch.nn as nn

from aoi_sentinel.models.lightweight import build_lightweight_encoder


class _LightweightClassifierImpl(nn.Module):
    def __init__(
        self,
        encoder_size: str = "small",     # "nano" | "small" | "pico"
        pretrained: bool = True,
        num_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = build_lightweight_encoder({
            "size": encoder_size,
            "pretrained": pretrained,
        })
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.embed_dim, num_classes)
        self.config = {
            "encoder_size": encoder_size,
            "pretrained": pretrained,
            "num_classes": num_classes,
            "dropout": dropout,
            "embed_dim": self.encoder.embed_dim,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        return self.head(self.dropout(feat))


# ---------------------------------------------------------------------------
# Cost-sensitive focal loss
# ---------------------------------------------------------------------------


class CostFocalLoss(nn.Module):
    """Focal loss with per-class weights derived from a cost matrix.

    Standard focal loss helps when the class imbalance is large. We extend
    it with `class_weights` set proportional to the row-sum of the cost
    matrix so the gradient is biased toward avoiding the catastrophic class
    (TRUE_DEFECT misclassification = escape).
    """

    def __init__(self, gamma: float = 2.0, class_weights: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.ones(2),
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_p = torch.log_softmax(logits, dim=-1)
        log_pt = log_p.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        focal = (1.0 - pt).pow(self.gamma)
        weights = self.class_weights[target]
        return -(focal * weights * log_pt).mean()


def class_weights_from_cost(cost_matrix) -> torch.Tensor:
    """Class weight ∝ Σ_a cost[label, a]  (= "how badly do we mis-handle this label").

    For our default automotive matrix (escape=1000, fc=1, op=5), this gives
    weights ≈ [6, 1005] → TRUE_DEFECT weighted ~170× more than FALSE_CALL.
    Capped to a max ratio of 50:1 to keep training stable.
    """
    import numpy as np
    cm = np.asarray(cost_matrix, dtype=float)
    raw = cm.sum(axis=1)
    raw = raw / raw.min()
    raw = np.minimum(raw, 50.0)
    return torch.tensor(raw, dtype=torch.float32)
