"""torch.utils.data.Dataset implementation, separated for lazy import."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from aoi_sentinel.runtime.label_queue import LabelRecord


_LABEL_TO_INT = {"FALSE_CALL": 0, "TRUE_DEFECT": 1}


class _LabelDatasetImpl(Dataset):
    def __init__(
        self,
        records: list[LabelRecord],
        image_root: str | Path = ".",
        roi_size: int = 224,
    ) -> None:
        self.records = [r for r in records if r.operator_label in _LABEL_TO_INT]
        self.image_root = Path(image_root)
        self.roi_size = roi_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        path = Path(r.image_path)
        if not path.is_absolute():
            path = self.image_root / path
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.roi_size, self.roi_size), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        # Standard ImageNet normalisation
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        x = (x - mean) / std
        y = _LABEL_TO_INT[r.operator_label]
        return x, y, {"board_id": r.board_id, "ref_des": r.ref_des}
