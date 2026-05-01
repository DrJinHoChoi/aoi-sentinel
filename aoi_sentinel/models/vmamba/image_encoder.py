"""MambaVision image encoder.

We pick MambaVision over plain VMamba because it (a) is timm-registered,
(b) has clean HuggingFace-hosted ImageNet weights across T/S/B/L sizes,
(c) hybrid Mamba + late self-attention transfers more cleanly under
distribution shift, and (d) ONNX-exports cleanly for edge deployment.

If MambaVision under-performs after Phase 0 pretraining, swap to VMamba V2 —
the rest of the system uses only the (B, D) embedding contract.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

ModelSize = Literal["tiny", "small", "base", "large"]


class ImageEncoder(nn.Module):
    """Embed an ROI image to a fixed-dim vector.

    Output shape: (B, embed_dim)
    """

    def __init__(
        self,
        size: ModelSize = "tiny",
        pretrained: bool = True,
        embed_dim: int | None = None,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        import timm

        timm_name = {
            "tiny": "mambavision_tiny_1k",
            "small": "mambavision_small_1k",
            "base": "mambavision_base_1k",
            "large": "mambavision_large_1k",
        }[size]
        self.backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features
        self.feat_dim = feat_dim

        if embed_dim is None or embed_dim == feat_dim:
            self.proj = nn.Identity()
            self.embed_dim = feat_dim
        else:
            self.proj = nn.Linear(feat_dim, embed_dim)
            self.embed_dim = embed_dim

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.proj(feat)


def build_image_encoder(cfg: dict) -> ImageEncoder:
    return ImageEncoder(
        size=cfg.get("size", "tiny"),
        pretrained=cfg.get("pretrained", True),
        embed_dim=cfg.get("embed_dim"),
        freeze_backbone=cfg.get("freeze_backbone", False),
    )
