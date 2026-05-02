"""Inferencer — load a checkpoint and emit (action, confidence) per ROI.

Used by `runtime/edge.py`. Implements Chow's reject rule on top of the
binary classifier so the engine emits the 3-action {DEFECT, PASS, ESCALATE}
without retraining for the third option.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np

Action = Literal["DEFECT", "PASS", "ESCALATE"]


class Inferencer:
    """Lazy-loaded torch inferencer.

    threshold     P(TRUE_DEFECT) above this → DEFECT
    abstain_band  if |P - threshold| < band → ESCALATE
                  (asymmetric: shrink the lower band when escapes are
                   dramatically more costly than false calls)
    """

    def __init__(
        self,
        weights_path: str | Path,
        config_path: str | Path | None = None,
        threshold: float = 0.30,         # below 0.5 because escape is costly
        abstain_band: float = 0.10,
        device: str | None = None,
    ) -> None:
        import torch

        self.weights_path = Path(weights_path)
        self.config_path = Path(config_path) if config_path else None
        self.threshold = float(threshold)
        self.abstain_band = float(abstain_band)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        ckpt = torch.load(self.weights_path, map_location=self.device, weights_only=False)
        cfg = ckpt.get("config", {})

        from aoi_sentinel.models.classifier._impl import _LightweightClassifierImpl
        self.model = _LightweightClassifierImpl(
            encoder_size=cfg.get("encoder_size", "small"),
            pretrained=False,                     # weights override pretrained
            num_classes=cfg.get("num_classes", 2),
            dropout=cfg.get("dropout", 0.0),     # eval-time dropout off
        ).to(self.device).eval()
        self.model.load_state_dict(ckpt["state_dict"])

        self._roi_size = 224

    @classmethod
    def from_handle(cls, handle, **kwargs) -> "Inferencer":
        """Construct from a `runtime.model_registry.ModelHandle`."""
        return cls(handle.weights_path, handle.config_path, **kwargs)

    def __call__(self, image: np.ndarray, height_map=None) -> tuple[Action, float]:
        import torch
        if image.ndim != 3 or image.shape[2] not in (1, 3):
            raise ValueError(f"expected HWC image, got shape {image.shape}")

        x = self._preprocess(image)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        p_true_defect = float(probs[1])
        action = self._decide(p_true_defect)
        # confidence on the chosen call (not always = max prob — for ESCALATE
        # we report distance to threshold so the UI can rank borderline cases)
        if action == "DEFECT":
            confidence = p_true_defect
        elif action == "PASS":
            confidence = 1.0 - p_true_defect
        else:
            confidence = 1.0 - abs(p_true_defect - self.threshold) / max(self.abstain_band, 1e-6)
            confidence = float(min(max(confidence, 0.0), 1.0))
        return action, confidence

    # ------------------------------------------------------------------ helpers

    def _preprocess(self, image: np.ndarray):
        import cv2
        import torch

        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        img = cv2.resize(image, (self._roi_size, self._roi_size), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return ((x - mean) / std).unsqueeze(0).to(self.device)

    def _decide(self, p: float) -> Action:
        hi = self.threshold + self.abstain_band
        lo = self.threshold - self.abstain_band
        if p > hi:
            return "DEFECT"
        if p < lo:
            return "PASS"
        return "ESCALATE"
