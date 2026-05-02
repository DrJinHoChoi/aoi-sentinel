"""Evaluation pipeline.

Karpathy principle: **eval first, train second.** This module exists before any
model training is meaningful, because without these curves we cannot tell
whether a new model improves over the incumbent or just looks shinier.

What this produces (every time, on every held-out stream):

    1. Cumulative-cost trajectory          — the headline economic number
    2. Risk-coverage (RC) curve + AURC     — selective-classification quality
    3. Cost curve (Drummond & Holte 2006)  — performance across cost ratios
    4. Daily / windowed false-call rate    — has it actually gone down?
    5. Daily escape rate                   — has it stayed at zero?
    6. Operator workload (ESCALATE rate)   — are we asking too much?
    7. Baseline-vs-engine delta            — must beat baseline, period.

The first three are scalar/curve metrics. The last four are time series we
plot to verify the system is improving and not just averaging better.
"""
from aoi_sentinel.eval.baseline import (
    selective_threshold_baseline,
    vendor_only_baseline,
)
from aoi_sentinel.eval.cost_curves import (
    aurc,
    cost_curve,
    expected_cost,
    risk_coverage_curve,
)
from aoi_sentinel.eval.metrics import AOIScore, score
from aoi_sentinel.eval.runner import EvalRecord, EvalReport, run_eval
from aoi_sentinel.eval.tracker import (
    FalseCallTracker,
    daily_metrics,
    windowed_metrics,
)

__all__ = [
    "AOIScore",
    "EvalRecord",
    "EvalReport",
    "FalseCallTracker",
    "aurc",
    "cost_curve",
    "daily_metrics",
    "expected_cost",
    "risk_coverage_curve",
    "run_eval",
    "score",
    "selective_threshold_baseline",
    "vendor_only_baseline",
    "windowed_metrics",
]
