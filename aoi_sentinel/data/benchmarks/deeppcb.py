"""DeepPCB loader.

Layout (Tang et al., GitHub `tangsanli5201/DeepPCB`):
    <root>/PCBData/group<NN>/<NN>/<id>_temp.jpg   — defect-free template
    <root>/PCBData/group<NN>/<NN>/<id>_test.jpg   — image with defects
    <root>/PCBData/group<NN>/<NN>/<id>.txt        — bounding boxes + class

We use the template/test pair structure directly as a false-call simulator:
    template image → label 0  (OK part)
    test image     → label 1  (defective part)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from aoi_sentinel.data.benchmarks.common import simulate_saki_calls, stack_images


def load_deeppcb(
    root: str | Path,
    size: int = 224,
    false_call_rate: float = 0.3,
    seed: int = 0,
):
    root = Path(root)
    pcb_data = root / "PCBData"
    if not pcb_data.exists():
        raise FileNotFoundError(f"DeepPCB layout not found under {root} — run scripts/download_deeppcb.py")

    paths: list[Path] = []
    labels: list[int] = []

    for sub in sorted(pcb_data.glob("group*")):
        for inner in sorted(sub.iterdir()):
            if not inner.is_dir():
                continue
            for temp in sorted(inner.glob("*_temp.jpg")):
                test = temp.with_name(temp.name.replace("_temp.jpg", "_test.jpg"))
                if not test.exists():
                    continue
                paths.append(temp)
                labels.append(0)
                paths.append(test)
                labels.append(1)

    images = stack_images(paths, size=size)
    labels_arr = np.asarray(labels, dtype=np.int64)
    saki_calls = simulate_saki_calls(labels_arr, false_call_rate=false_call_rate, seed=seed)
    return images, labels_arr, saki_calls
