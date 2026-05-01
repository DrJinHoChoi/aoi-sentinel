"""PyTorch dataset for the 2D false-call classifier.

Each sample is a cropped ROI around a Saki-flagged defect. The label is
binary: 1 = true defect (operator confirmed), 0 = false call.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # train extras not installed
    torch = None  # type: ignore
    Dataset = object  # type: ignore


class SakiROIDataset(Dataset):
    """ROI crops around Saki-flagged defects.

    Index parquet must have columns:
        image_2d_path, bbox_xyxy (list[int]), label (0/1),
        saki_defect_type (str), board_id (str)
    """

    def __init__(
        self,
        index_path: str | Path,
        roi_size: int = 224,
        pad_ratio: float = 0.25,
        transform: Any = None,
    ) -> None:
        self.df = pd.read_parquet(index_path)
        self.roi_size = roi_size
        self.pad_ratio = pad_ratio
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        import cv2

        row = self.df.iloc[idx]
        img = cv2.imread(str(row["image_2d_path"]), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(row["image_2d_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        x1, y1, x2, y2 = row["bbox_xyxy"]
        w, h = x2 - x1, y2 - y1
        pad = int(max(w, h) * self.pad_ratio)
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(img.shape[1], x2 + pad), min(img.shape[0], y2 + pad)
        roi = img[y1:y2, x1:x2]

        roi = cv2.resize(roi, (self.roi_size, self.roi_size), interpolation=cv2.INTER_AREA)

        if self.transform is not None:
            roi = self.transform(image=roi)["image"]
        else:
            roi = torch.from_numpy(roi.transpose(2, 0, 1)).float() / 255.0

        label = int(row["label"])
        return roi, label, {"defect_type": row["saki_defect_type"], "board_id": row["board_id"]}


def stratified_split(
    index_path: str | Path,
    out_dir: str | Path,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, int]:
    """Stratify by (saki_defect_type, label) and split board-wise to avoid leakage."""
    df = pd.read_parquet(index_path)
    rng = np.random.default_rng(seed)

    boards = df["board_id"].unique()
    rng.shuffle(boards)
    n = len(boards)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test_boards = set(boards[:n_test])
    val_boards = set(boards[n_test : n_test + n_val])

    def split_of(b: str) -> str:
        if b in test_boards:
            return "test"
        if b in val_boards:
            return "val"
        return "train"

    df["split"] = df["board_id"].map(split_of)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        sub.to_parquet(out_dir / f"{split}.parquet", index=False)
        counts[split] = len(sub)
    return counts
