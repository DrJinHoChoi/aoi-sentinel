"""VisA loader (Amazon Science).

Layout (after download, see scripts/download_visa.py):
    <root>/
        pcb1/
            Data/Images/Anomaly/*.JPG     — defective
            Data/Images/Normal/*.JPG      — OK
        pcb2/...  pcb3/...  pcb4/...

We collapse all four PCB classes into a single binary stream:
    label = 0 (FALSE_CALL surrogate)  for OK
    label = 1 (TRUE_DEFECT)           for Anomaly
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from aoi_sentinel.data.benchmarks.common import simulate_saki_calls, stack_images

PCB_CLASSES = ("pcb1", "pcb2", "pcb3", "pcb4")


def load_visa(
    root: str | Path,
    size: int = 224,
    classes: tuple[str, ...] = PCB_CLASSES,
    false_call_rate: float = 0.3,
    seed: int = 0,
):
    root = Path(root)
    paths: list[Path] = []
    labels: list[int] = []

    for cls in classes:
        cls_root = root / cls / "Data" / "Images"
        for sub in ("Normal", "Anomaly"):
            d = cls_root / sub
            if not d.exists():
                continue
            for p in sorted(d.glob("*.JPG")):
                paths.append(p)
                labels.append(0 if sub == "Normal" else 1)

    if not paths:
        raise FileNotFoundError(f"No VisA images under {root} — run scripts/download_visa.py first")

    images = stack_images(paths, size=size)
    labels_arr = np.asarray(labels, dtype=np.int64)
    saki_calls = simulate_saki_calls(labels_arr, false_call_rate=false_call_rate, seed=seed)
    return images, labels_arr, saki_calls
