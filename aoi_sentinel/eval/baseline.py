"""Reference baselines.

Karpathy: every claim of "our model is better" must come with a baseline
comparison. We pre-define the baselines that any candidate must beat.

  vendor_only_baseline    — what would happen if we did nothing (Saki/Koh
                            Young raw call accepted as truth). Sets the
                            ceiling on cost.
  selective_threshold     — a calibrated supervised classifier with a
                            fixed reject threshold. The strong baseline
                            that any RL policy must beat to justify itself.
                            Most "RL" contributions on contextual bandit
                            problems quietly fail this one — we surface it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class BaselineResult:
    name: str
    actions: np.ndarray         # (N,) of "DEFECT" | "PASS" | "ESCALATE"
    confidences: np.ndarray     # (N,) float in [0, 1]


# ---------------------------------------------------------------------------
# Vendor-only baseline
# ---------------------------------------------------------------------------


def vendor_only_baseline(vendor_calls: Sequence[str]) -> BaselineResult:
    """The "do-nothing" baseline.

    Vendor flagged DEFECT  → we keep DEFECT (operator re-checks all of them)
    Vendor flagged PASS    → we keep PASS
    Vendor flagged UNKNOWN → ESCALATE
    """
    actions = np.empty(len(vendor_calls), dtype=object)
    confidences = np.zeros(len(vendor_calls), dtype=np.float32)
    for i, vc in enumerate(vendor_calls):
        if vc == "DEFECT":
            actions[i] = "DEFECT"; confidences[i] = 0.5
        elif vc == "PASS":
            actions[i] = "PASS"; confidences[i] = 0.5
        else:
            actions[i] = "ESCALATE"; confidences[i] = 0.0
    return BaselineResult(name="vendor_only", actions=actions, confidences=confidences)


# ---------------------------------------------------------------------------
# Selective-threshold baseline  (supervised classifier + Chow rule)
# ---------------------------------------------------------------------------


def selective_threshold_baseline(
    posteriors: Sequence[float],
    threshold: float = 0.5,
    abstain_band: float = 0.1,
) -> BaselineResult:
    """Strong supervised baseline: threshold the calibrated posterior.

    posteriors[i] = P(TRUE_DEFECT | x_i)  in [0, 1]

    Decision rule (Chow's optimal reject rule, simplified):
        p > threshold + band      → DEFECT
        p < threshold - band      → PASS
        otherwise                 → ESCALATE

    The default `threshold=0.5` corresponds to symmetric costs; for our
    asymmetric setup (escape ~1000× FC) the right threshold is much lower.
    Tune it on a hold-out using `eval.cost_curves.expected_cost`.
    """
    p = np.asarray(posteriors, dtype=np.float32)
    n = len(p)
    actions = np.empty(n, dtype=object)
    hi, lo = threshold + abstain_band, threshold - abstain_band
    for i in range(n):
        if p[i] > hi:
            actions[i] = "DEFECT"
        elif p[i] < lo:
            actions[i] = "PASS"
        else:
            actions[i] = "ESCALATE"
    confidences = np.where(p > 0.5, p, 1.0 - p).astype(np.float32)
    return BaselineResult(name="selective_threshold", actions=actions, confidences=confidences)
