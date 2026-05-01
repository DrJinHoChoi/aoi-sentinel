"""Trajectory-level metrics for the NPI online setting.

The Saki false-call problem is judged over time, not at a single threshold.
We track the running cost, escape rate, false-call rate, and ESCALATE rate as
the agent learns — these are the curves that justify shipping vs. not shipping.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aoi_sentinel.sim.cost import (
    ACTION_DEFECT,
    ACTION_ESCALATE,
    ACTION_PASS,
    LABEL_TRUE_DEFECT,
)


@dataclass
class NpiTrajectoryStats:
    cumulative_cost: float
    escape_rate: float
    false_call_rate: float
    escalate_rate: float
    n_steps: int


def summarize(actions: np.ndarray, labels: np.ndarray, costs: np.ndarray) -> NpiTrajectoryStats:
    n = len(actions)
    escapes = ((labels == LABEL_TRUE_DEFECT) & (actions == ACTION_PASS)).sum()
    false_call_kept = ((labels == 0) & (actions == ACTION_DEFECT)).sum()
    n_false_call = (labels == 0).sum()
    n_true_defect = (labels == LABEL_TRUE_DEFECT).sum()
    return NpiTrajectoryStats(
        cumulative_cost=float(costs.sum()),
        escape_rate=float(escapes) / max(int(n_true_defect), 1),
        false_call_rate=float(false_call_kept) / max(int(n_false_call), 1),
        escalate_rate=float((actions == ACTION_ESCALATE).sum()) / n,
        n_steps=n,
    )
