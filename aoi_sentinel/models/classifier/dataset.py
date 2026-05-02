"""Dataset: feeds the classifier from a `LabelQueue` SQLite stream.

The on-prem trainer pulls labels from each edge's queue, builds train +
holdout splits, and trains. We deliberately split **board-wise** to avoid
leakage: components from the same board are correlated, so they must
stay together in the same split.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from aoi_sentinel.runtime.label_queue import LabelRecord


# ---------------------------------------------------------------------------
# Filtering / split helpers — pure stdlib, torch-free, runs anywhere
# ---------------------------------------------------------------------------


@dataclass
class Split:
    train: list[LabelRecord]
    holdout: list[LabelRecord]


def board_wise_split(
    records: list[LabelRecord],
    holdout_fraction: float = 0.15,
    seed: int = 42,
) -> Split:
    """Split a label stream by `board_id` so no board straddles train/holdout."""
    if not records:
        return Split(train=[], holdout=[])
    rng = np.random.default_rng(seed)
    boards = sorted({r.board_id for r in records})
    rng.shuffle(boards)
    n_holdout = max(1, int(len(boards) * holdout_fraction)) if len(boards) > 1 else 0
    holdout_boards = set(boards[:n_holdout])
    train, holdout = [], []
    for r in records:
        (holdout if r.board_id in holdout_boards else train).append(r)
    return Split(train=train, holdout=holdout)


def filter_labelable(records: Iterable[LabelRecord]) -> list[LabelRecord]:
    """Keep only rows where the operator gave a definitive label.
    UNSURE rows are excluded from supervised training but kept for the
    safety gate's holdout scoring."""
    return [r for r in records if r.operator_label in {"TRUE_DEFECT", "FALSE_CALL"}]


# ---------------------------------------------------------------------------
# torch.utils.data.Dataset — lazy, only built when training actually runs
# ---------------------------------------------------------------------------


class LabelDataset:
    """Lazy wrapper. Module-level import does not require torch."""

    def __new__(cls, *args, **kwargs):
        from aoi_sentinel.models.classifier._dataset_impl import _LabelDatasetImpl
        return _LabelDatasetImpl(*args, **kwargs)


def build_dataset(records: list[LabelRecord], image_root: str | Path = ".", roi_size: int = 224):
    """Convenience constructor with the most-common defaults."""
    return LabelDataset(records=records, image_root=image_root, roi_size=roi_size)
