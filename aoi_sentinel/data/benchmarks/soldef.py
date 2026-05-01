"""SolDef_AI loader (Mendeley Data, 2024).

Solder-joint level defects (good / insufficient / excess / bridge). We map:
    good  → label 0
    other → label 1

Exact layout depends on the Mendeley export — adjust `_walk` once the dump arrives.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from aoi_sentinel.data.benchmarks.common import simulate_saki_calls, stack_images


def load_soldef(
    root: str | Path,
    size: int = 224,
    false_call_rate: float = 0.3,
    seed: int = 0,
):
    root = Path(root)
    paths, labels = _walk(root)
    if not paths:
        raise FileNotFoundError(f"No SolDef_AI images under {root}")
    images = stack_images(paths, size=size)
    labels_arr = np.asarray(labels, dtype=np.int64)
    saki_calls = simulate_saki_calls(labels_arr, false_call_rate=false_call_rate, seed=seed)
    return images, labels_arr, saki_calls


def _walk(root: Path) -> tuple[list[Path], list[int]]:
    paths: list[Path] = []
    labels: list[int] = []
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        is_good = cls_dir.name.lower() in {"good", "ok", "normal"}
        for img in sorted(cls_dir.glob("*.[pj][np]g")):
            paths.append(img)
            labels.append(0 if is_good else 1)
    return paths, labels
