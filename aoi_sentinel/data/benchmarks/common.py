"""Shared utilities for benchmark loaders."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def stack_images(paths: list[Path], size: int = 224) -> np.ndarray:
    """Load and resize images into a single (N, H, W, 3) uint8 array."""
    import cv2

    out = np.empty((len(paths), size, size, 3), dtype=np.uint8)
    for i, p in enumerate(paths):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        out[i] = img
    return out


def simulate_saki_calls(labels: np.ndarray, false_call_rate: float = 0.3, seed: int = 0) -> np.ndarray:
    """Simulate Saki's noisy NG flagging.

    The Saki rule-based call has high recall on real defects but emits ~30% false calls.
    We simulate by flagging:
      - all true defects (recall ≈ 1.0)
      - plus a fraction of OK parts equal to false_call_rate
    Returned mask is 1 where Saki flags NG.
    """
    rng = np.random.default_rng(seed)
    saki = labels.copy()
    ok_idx = np.where(labels == 0)[0]
    n_fc = int(len(ok_idx) * false_call_rate)
    saki[rng.choice(ok_idx, size=n_fc, replace=False)] = 1
    return saki.astype(np.int64)
