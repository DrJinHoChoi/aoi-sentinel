"""AOI-specific metrics.

Two metrics matter most in production:

  * false_call_reduction:   how many Saki false calls we correctly flip to PASS.
  * escape_rate:            fraction of TRUE defects we incorrectly let through.

Escape rate must stay near zero — letting a true defect through ships a bad
board, which is far more costly than re-inspecting a false call.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AOIScore:
    n: int
    true_defect_recall: float       # of all true defects, how many we kept flagged
    false_call_reduction: float     # of all false calls, how many we cleared
    escape_rate: float              # 1 - true_defect_recall, surfaced for visibility
    accuracy: float


def score(y_true: list[int], y_pred: list[int]) -> AOIScore:
    """y_true / y_pred: 1 = TRUE_DEFECT, 0 = FALSE_CALL."""
    if len(y_true) != len(y_pred):
        raise ValueError("length mismatch")
    n = len(y_true)
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fc_reduction = tn / (tn + fp) if (tn + fp) else 0.0
    return AOIScore(
        n=n,
        true_defect_recall=recall,
        false_call_reduction=fc_reduction,
        escape_rate=1.0 - recall,
        accuracy=(tp + tn) / n if n else 0.0,
    )
