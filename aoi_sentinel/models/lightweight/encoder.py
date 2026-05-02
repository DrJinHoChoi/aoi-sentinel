"""Lightweight image encoder for edge inference.

Drop-in for `models.vmamba.image_encoder.ImageEncoder` — same forward signature
and `embed_dim` attribute, much smaller weights.

Knowledge-distillation script (planned in Phase 2) trains this against the
heavyweight teacher (MambaVision-T) on the trainer so the edge model stays
within ~1 pp of the teacher's accuracy.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

LightweightSize = Literal["nano", "small", "pico"]
"""
nano  — MobileNetV3-Small         (≈ 2.5M params, Jetson Nano POC)
small — MobileNetV3-Large         (≈ 5.5M params, Orin Nano)
pico  — ConvNeXt-Pico (timm)      (≈ 9M params,   Orin NX or better)
"""


class LightweightEncoder(nn.Module):
    def __init__(
        self,
        size: LightweightSize = "small",
        pretrained: bool = True,
        embed_dim: int | None = None,
    ) -> None:
        super().__init__()
        import timm

        timm_name = {
            "nano": "mobilenetv3_small_100",
            "small": "mobilenetv3_large_100",
            "pico": "convnext_pico",
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x))


def build_lightweight_encoder(cfg: dict) -> LightweightEncoder:
    return LightweightEncoder(
        size=cfg.get("size", "small"),
        pretrained=cfg.get("pretrained", True),
        embed_dim=cfg.get("embed_dim"),
    )
