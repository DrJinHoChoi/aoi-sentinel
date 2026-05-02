"""Structure tests for the classifier + trainer wiring.

Pure-Python tests that verify the data-prep / split / scoring logic without
touching torch. The torch-dependent training loop is exercised by the
Colab notebook + (future) integration tests on the trainer box.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from aoi_sentinel.models.classifier.dataset import (
    Split,
    board_wise_split,
    filter_labelable,
)
from aoi_sentinel.runtime.label_queue import LabelRecord


def _rec(board: str, ref: str, label: str, t: datetime) -> LabelRecord:
    return LabelRecord(
        board_id=board, ref_des=ref, vendor="generic_csv", line_id="L1",
        timestamp=t, image_path=f"boards/{board}/{ref}.jpg",
        height_map_path=None, vendor_call="DEFECT", vendor_defect_type=None,
        engine_action="ESCALATE", engine_confidence=0.5,
        operator_label=label, operator_id="op-1", model_version="v0",
    )


# ----------------------------------------------------------------- filtering

def test_filter_labelable_drops_unsure():
    t = datetime.now(timezone.utc)
    rows = [
        _rec("B1", "C1", "TRUE_DEFECT", t),
        _rec("B1", "C2", "UNSURE", t),
        _rec("B2", "C3", "FALSE_CALL", t),
    ]
    kept = filter_labelable(rows)
    assert len(kept) == 2
    assert all(r.operator_label in {"TRUE_DEFECT", "FALSE_CALL"} for r in kept)


# ---------------------------------------------------------------------- split

def test_board_wise_split_preserves_boards():
    t = datetime.now(timezone.utc)
    rows = []
    for b in range(10):
        for c in range(5):
            rows.append(_rec(f"B{b}", f"C{c}", "FALSE_CALL", t + timedelta(minutes=b*5+c)))

    split = board_wise_split(rows, holdout_fraction=0.2, seed=0)
    train_boards = {r.board_id for r in split.train}
    holdout_boards = {r.board_id for r in split.holdout}

    # No board straddles the split
    assert not (train_boards & holdout_boards)
    # Roughly 20% of boards in holdout
    assert 1 <= len(holdout_boards) <= 4


def test_board_wise_split_handles_empty():
    s = board_wise_split([], holdout_fraction=0.2)
    assert s.train == []
    assert s.holdout == []


def test_board_wise_split_single_board_keeps_in_train():
    t = datetime.now(timezone.utc)
    rows = [_rec("B0", f"C{i}", "FALSE_CALL", t) for i in range(5)]
    s = board_wise_split(rows, holdout_fraction=0.2)
    # Single board can't be split — must stay together
    assert (s.train and not s.holdout) or (s.holdout and not s.train)


# --------------------------------------------------- trainer data preparation

def test_trainer_prepare_data_temporal_split():
    """Most recent N labels reserved as holdout — never trained on."""
    from aoi_sentinel.runtime.trainer_server import _prepare_data
    from aoi_sentinel.runtime.label_queue import LabelQueue

    # In-memory queue using tmp file
    import tempfile
    from pathlib import Path
    tmp = tempfile.mkdtemp()
    q = LabelQueue(Path(tmp) / "labels.db")

    t0 = datetime(2026, 4, 1, tzinfo=timezone.utc)
    for i in range(50):
        q.append(_rec(f"B{i//5}", f"C{i%5}", "FALSE_CALL", t0 + timedelta(minutes=i)))

    train, holdout = _prepare_data([q], {q.db_path: 0}, holdout_size=10)
    assert len(holdout) == 10
    assert len(train) == 40
    # Holdout is the most recent
    assert all(h.timestamp >= t.timestamp for h in holdout for t in train[-10:])


def test_trainer_prepare_data_short_stream_all_holdout():
    """If we have fewer rows than holdout_size, all go to holdout."""
    from aoi_sentinel.runtime.trainer_server import _prepare_data
    from aoi_sentinel.runtime.label_queue import LabelQueue
    import tempfile
    from pathlib import Path

    tmp = tempfile.mkdtemp()
    q = LabelQueue(Path(tmp) / "labels.db")
    t0 = datetime(2026, 4, 1, tzinfo=timezone.utc)
    for i in range(5):
        q.append(_rec(f"B{i}", "C0", "FALSE_CALL", t0 + timedelta(minutes=i)))

    train, holdout = _prepare_data([q], {q.db_path: 0}, holdout_size=10)
    assert train == []
    assert len(holdout) == 5
