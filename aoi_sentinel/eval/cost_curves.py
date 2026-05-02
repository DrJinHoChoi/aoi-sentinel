"""Cost-asymmetric evaluation curves.

These are the right metrics for our problem — accuracy / F1 / AUROC are all
misleading when one error is 1000× more expensive than the other.

References
----------
- Drummond & Holte, "Cost Curves: An Improved Method for Visualizing
  Classifier Performance", Machine Learning 2006.
- El-Yaniv & Wiener, "On the Foundations of Noise-free Selective
  Classification", JMLR 2010.
- Geifman & El-Yaniv, "Selective Classification for Deep Neural Networks",
  NeurIPS 2017.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# Action ids — match aoi_sentinel.sim.cost
ACTION_DEFECT = 0
ACTION_PASS = 1
ACTION_ESCALATE = 2

LABEL_FALSE_CALL = 0
LABEL_TRUE_DEFECT = 1


# ---------------------------------------------------------------------------
# Expected cost
# ---------------------------------------------------------------------------


def expected_cost(
    labels: Sequence[int],
    actions: Sequence[int],
    cost_matrix: np.ndarray,
) -> float:
    """Mean per-decision cost under a 2x3 cost matrix [label, action]."""
    if len(labels) != len(actions):
        raise ValueError("labels / actions length mismatch")
    if not len(labels):
        return 0.0
    cm = np.asarray(cost_matrix)
    if cm.shape != (2, 3):
        raise ValueError(f"cost_matrix must be 2x3, got {cm.shape}")
    labels_arr = np.asarray(labels, dtype=int)
    actions_arr = np.asarray(actions, dtype=int)
    return float(cm[labels_arr, actions_arr].mean())


# ---------------------------------------------------------------------------
# Risk-coverage curve  (selective classification)
# ---------------------------------------------------------------------------


@dataclass
class RiskCoverage:
    """One run's risk-coverage curve.

    coverage[k] = fraction of decisions we accepted (didn't ESCALATE) at
                  threshold rank k
    risk[k]     = error rate among the accepted decisions at threshold rank k
    """

    coverage: np.ndarray
    risk: np.ndarray


def risk_coverage_curve(
    labels: Sequence[int],
    binary_predictions: Sequence[int],
    confidences: Sequence[float],
) -> RiskCoverage:
    """Sweep an abstention threshold over `confidences`.

    `binary_predictions` is the model's hard {0=FALSE_CALL, 1=TRUE_DEFECT}
    call (no ESCALATE). At each threshold we abstain on the lowest-confidence
    decisions and measure error on the rest.

    Right tail = high coverage, lower threshold (we accept everything) →
                 risk equals raw error rate.
    Left tail  = low coverage, high threshold (we abstain on most) →
                 risk should approach 0 if confidence is calibrated.
    """
    labels_arr = np.asarray(labels, dtype=int)
    preds_arr = np.asarray(binary_predictions, dtype=int)
    conf_arr = np.asarray(confidences, dtype=float)
    n = len(labels_arr)
    if not (len(preds_arr) == len(conf_arr) == n):
        raise ValueError("length mismatch")
    if n == 0:
        return RiskCoverage(np.array([]), np.array([]))

    # Sort by descending confidence; we accept the top-k most confident.
    order = np.argsort(-conf_arr)
    sorted_correct = (labels_arr[order] == preds_arr[order]).astype(np.int64)
    cum_correct = np.cumsum(sorted_correct)
    k = np.arange(1, n + 1)
    coverage = k / n
    risk = 1.0 - cum_correct / k
    return RiskCoverage(coverage=coverage, risk=risk)


def aurc(rc: RiskCoverage) -> float:
    """Area under the risk-coverage curve (lower is better).

    AURC integrates risk over all coverage levels. A perfect calibrator
    that ranks all errors below all corrects achieves AURC=0; a random
    classifier gives AURC ≈ raw error rate.
    """
    if rc.coverage.size == 0:
        return 0.0
    return float(np.trapezoid(rc.risk, rc.coverage))


# ---------------------------------------------------------------------------
# Cost curve  (Drummond & Holte 2006)
# ---------------------------------------------------------------------------


def cost_curve(
    labels: Sequence[int],
    binary_predictions: Sequence[int],
    n_points: int = 101,
) -> tuple[np.ndarray, np.ndarray]:
    """Drummond-Holte cost curve.

    x-axis: probability-cost function PCF(+) ∈ [0, 1]
              = π · C_FN / (π · C_FN + (1−π) · C_FP)
            sweeping over both class prior π and cost ratio.
    y-axis: normalised expected cost NEC ∈ [0, 1]
              = FNR · PCF(+) + FPR · (1 − PCF(+))

    A perfect classifier sits on NEC=0. A trivial "always positive" or
    "always negative" sits on the diagonal y = 1 - x or y = x respectively.
    The classifier curve is the lower envelope across operating points.

    For our binary {FALSE_CALL=0, TRUE_DEFECT=1} setup, "positive" = TRUE_DEFECT.
    FN = true_defect predicted FALSE_CALL  (escape — bad)
    FP = false_call predicted TRUE_DEFECT  (false call — bad but cheap)
    """
    labels_arr = np.asarray(labels, dtype=int)
    preds_arr = np.asarray(binary_predictions, dtype=int)
    pos = labels_arr == LABEL_TRUE_DEFECT
    neg = ~pos
    fnr = float((preds_arr[pos] != LABEL_TRUE_DEFECT).mean()) if pos.any() else 0.0
    fpr = float((preds_arr[neg] != LABEL_FALSE_CALL).mean()) if neg.any() else 0.0

    pcf = np.linspace(0.0, 1.0, n_points)
    nec = fnr * pcf + fpr * (1.0 - pcf)
    return pcf, nec
