"""MambaVision image encoder with robust resolver and ConvNeXt fallback.

Why this is more than 5 lines around timm.create_model:
    The `mambavision` PyPI package's registration story keeps shifting
    between releases — sometimes it auto-registers with timm, sometimes
    it ships its own `create_model` API, sometimes the model name is
    `mambavision_tiny_1k` and sometimes `mamba_vision_T`. We try all
    known paths in order, and finally fall back to ConvNeXt-Tiny
    (the strong Karpathy baseline) so v0 ships even if MambaVision is
    unavailable in the runtime.

Resolution order:
    1. mambavision package's own `create_model` API
    2. `import mambavision` to trigger timm side-effect registration,
       then `timm.create_model` with current naming
    3. Older naming (`mamba_vision_T`)
    4. ConvNeXt-Tiny / Small / Base / Large fallback (no Mamba — but
       runs everywhere torch + timm runs)

The (B, embed_dim) output contract is identical in all cases, so the
rest of the system (policy, classifier, RL) is oblivious to which
backend was selected.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

ModelSize = Literal["tiny", "small", "base", "large"]


# Known MambaVision name conventions across releases.
_MV_NAMES_NEW = {
    "tiny":  "mambavision_tiny_1k",
    "small": "mambavision_small_1k",
    "base":  "mambavision_base_1k",
    "large": "mambavision_large_1k",
}
_MV_NAMES_LEGACY = {
    "tiny":  "mamba_vision_T",
    "small": "mamba_vision_S",
    "base":  "mamba_vision_B",
    "large": "mamba_vision_L",
}
_CONVNEXT_FALLBACK = {
    "tiny":  "convnext_tiny",
    "small": "convnext_small",
    "base":  "convnext_base",
    "large": "convnext_large",
}


def _build_backbone(size: ModelSize, pretrained: bool):
    """Try every known way to instantiate MambaVision; fall back to ConvNeXt."""
    import timm

    # Path 1: mambavision package's own create_model
    try:
        from mambavision import create_model as mv_create  # type: ignore

        for name in (_MV_NAMES_LEGACY[size], _MV_NAMES_NEW[size]):
            try:
                model = mv_create(name, pretrained=pretrained, num_classes=0)
                print(f"[image_encoder] using mambavision.create_model('{name}')")
                return model, "mambavision_native"
            except (ValueError, KeyError):
                continue
    except ImportError:
        pass

    # Path 2: import mambavision to register with timm, then timm.create_model
    try:
        import mambavision  # noqa: F401  — registration side-effect
    except ImportError:
        pass

    for name in (_MV_NAMES_NEW[size], _MV_NAMES_LEGACY[size]):
        try:
            model = timm.create_model(name, pretrained=pretrained, num_classes=0)
            print(f"[image_encoder] using timm.create_model('{name}')")
            return model, "mambavision_timm"
        except RuntimeError:
            continue

    # Path 3: ConvNeXt fallback — the strong Karpathy baseline. Always works.
    name = _CONVNEXT_FALLBACK[size]
    print(f"[image_encoder] MambaVision unavailable; falling back to ConvNeXt → '{name}'")
    return timm.create_model(name, pretrained=pretrained, num_classes=0), "convnext_fallback"


class ImageEncoder(nn.Module):
    """Embed an ROI image to a fixed-dim vector.  Output: (B, embed_dim)."""

    def __init__(
        self,
        size: ModelSize = "tiny",
        pretrained: bool = True,
        embed_dim: int | None = None,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone, self.backend_used = _build_backbone(size, pretrained)
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
