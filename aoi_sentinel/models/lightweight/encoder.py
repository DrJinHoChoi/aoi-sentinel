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

LightweightSize = Literal["nano", "small", "pico", "tiny", "base", "large"]
"""
Native names:
  nano  — MobileNetV3-Small         (≈ 2.5M params, Jetson Nano POC)
  small — MobileNetV3-Large         (≈ 5.5M params, Orin Nano)
  pico  — ConvNeXt-Pico (timm)      (≈ 9M params,   Orin NX or better)

Aliases for MambaVision-style size names (so the same config dict can
drive either backbone — the policy never has to know which encoder it got):
  tiny  → small  (≈ 5.5M params)
  base  → pico   (≈ 9M params)
  large → pico   (capped at pico — anything bigger isn't "lightweight")
"""

# timm model name + optional alias
_NAME_MAP = {
    "nano":  "mobilenetv3_small_100",
    "small": "mobilenetv3_large_100",
    "pico":  "convnext_pico",
    # MambaVision-style aliases — graceful when configs flip between backbones
    "tiny":  "mobilenetv3_large_100",
    "base":  "convnext_pico",
    "large": "convnext_pico",
}


class LightweightEncoder(nn.Module):
    def __init__(
        self,
        size: LightweightSize = "small",
        pretrained: bool = True,
        embed_dim: int | None = None,
    ) -> None:
        super().__init__()
        import timm

        if size not in _NAME_MAP:
            raise ValueError(f"unknown lightweight size {size!r}; valid: {list(_NAME_MAP)}")
        timm_name = _NAME_MAP[size]

        self.backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=0)
        feat_dim = _probe_output_dim(self.backbone)
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


def _probe_output_dim(model: nn.Module, image_size: int = 224) -> int:
    """Run a dummy forward to determine the actual output dim.

    `model.num_features` in timm is reliable for some architectures and
    misleading for others (notably MobileNetV3, where `num_features`
    reports the pre-conv-head value but the actual `forward()` output
    is post-conv-head). We trust the actual tensor shape.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out = model(torch.zeros(1, 3, image_size, image_size))
    finally:
        model.train(was_training)
    if out.dim() == 2:
        return int(out.shape[-1])
    # Pool a 4-D feature map (B, C, H, W) to a vector.
    if out.dim() == 4:
        return int(out.shape[1])
    raise RuntimeError(f"unexpected encoder output rank {out.dim()}: shape={tuple(out.shape)}")
