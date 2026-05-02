"""End-to-end eval runner.

Single function: read a labelled stream → produce the report dict that
contains every metric defined in this package. Everything downstream
(plotting, dashboard, safety gate) reads this one structure.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from aoi_sentinel.eval.baseline import vendor_only_baseline
from aoi_sentinel.eval.cost_curves import (
    LABEL_FALSE_CALL,
    LABEL_TRUE_DEFECT,
    aurc,
    cost_curve,
    expected_cost,
    risk_coverage_curve,
)
from aoi_sentinel.eval.tracker import daily_metrics, windowed_metrics

# Action / label string ↔ int conversions
_ACTION_TO_INT = {"DEFECT": 0, "PASS": 1, "ESCALATE": 2}
_LABEL_TO_INT = {"FALSE_CALL": LABEL_FALSE_CALL, "TRUE_DEFECT": LABEL_TRUE_DEFECT}


# ---------------------------------------------------------------------------
# I/O records
# ---------------------------------------------------------------------------


@dataclass
class EvalRecord:
    """One labelled decision — the unit of evaluation."""

    timestamp: datetime
    vendor_call: str            # "PASS" | "DEFECT" | "UNKNOWN"
    engine_action: str          # "PASS" | "DEFECT" | "ESCALATE"
    engine_confidence: float
    label: str                  # "TRUE_DEFECT" | "FALSE_CALL"


@dataclass
class EvalReport:
    """All numbers a stakeholder would ask for."""

    n_total: int
    n_labeled: int
    cost_matrix: list[list[float]]

    engine: dict = field(default_factory=dict)
    baseline_vendor_only: dict = field(default_factory=dict)
    delta: dict = field(default_factory=dict)

    daily: list[dict] = field(default_factory=list)
    windowed: dict = field(default_factory=dict)

    aurc_engine: float = 0.0
    cost_curve: dict = field(default_factory=dict)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# Default cost matrix — kept in sync with sim.cost.CostMatrix defaults
# ---------------------------------------------------------------------------


def _default_cost_matrix() -> np.ndarray:
    cm = np.zeros((2, 3), dtype=np.float32)
    # [label, action] — action: 0=DEFECT, 1=PASS, 2=ESCALATE
    cm[LABEL_FALSE_CALL, 0] = 1.0      # false call (rework cost)
    cm[LABEL_FALSE_CALL, 1] = 0.0      # correctly cleared
    cm[LABEL_FALSE_CALL, 2] = 5.0      # operator review
    cm[LABEL_TRUE_DEFECT, 0] = 0.0     # correctly flagged
    cm[LABEL_TRUE_DEFECT, 1] = 1000.0  # ESCAPE — the catastrophe
    cm[LABEL_TRUE_DEFECT, 2] = 5.0     # operator review
    return cm


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_eval(
    records: Iterable[EvalRecord],
    cost_matrix: np.ndarray | None = None,
    daily: bool = True,
    windowed_window: int = 1_000,
    windowed_step: int = 100,
) -> EvalReport:
    """Compute every metric on a labelled stream."""
    cm = cost_matrix if cost_matrix is not None else _default_cost_matrix()
    recs = list(records)

    n_total = len(recs)
    labeled = [r for r in recs if r.label in _LABEL_TO_INT]
    n_labeled = len(labeled)
    if not labeled:
        return EvalReport(n_total=n_total, n_labeled=0, cost_matrix=cm.tolist())

    labels = np.array([_LABEL_TO_INT[r.label] for r in labeled], dtype=int)
    engine_actions = np.array([_ACTION_TO_INT[r.engine_action] for r in labeled], dtype=int)
    engine_conf = np.array([r.engine_confidence for r in labeled], dtype=np.float32)
    vendor_calls = [r.vendor_call for r in labeled]

    # ---- engine
    engine_cost = expected_cost(labels, engine_actions, cm)
    engine_summary = _scalar_summary(labels, engine_actions, engine_cost)

    # ---- baseline (vendor-only)
    bl = vendor_only_baseline(vendor_calls)
    bl_actions = np.array([_ACTION_TO_INT[a] for a in bl.actions], dtype=int)
    bl_cost = expected_cost(labels, bl_actions, cm)
    bl_summary = _scalar_summary(labels, bl_actions, bl_cost)

    # ---- delta (engine vs baseline)
    delta = {
        "expected_cost_drop": bl_summary["expected_cost"] - engine_summary["expected_cost"],
        "false_call_drop_pp": bl_summary["false_call_rate"] - engine_summary["false_call_rate"],
        "escape_drop_pp": bl_summary["escape_rate"] - engine_summary["escape_rate"],
        "escalate_increase_pp": engine_summary["escalate_rate"] - bl_summary["escalate_rate"],
    }

    # ---- AURC (only for non-ESCALATE decisions, where we made a hard call)
    decisive = engine_actions != 2  # not ESCALATE
    if decisive.any():
        rc = risk_coverage_curve(
            labels[decisive],
            np.where(engine_actions[decisive] == 0, 1, 0),  # DEFECT→1=TRUE_DEFECT
            engine_conf[decisive],
        )
        engine_aurc = aurc(rc)
    else:
        engine_aurc = 0.0

    # ---- cost curve (Drummond-Holte)
    if decisive.any():
        pcf, nec = cost_curve(
            labels[decisive],
            np.where(engine_actions[decisive] == 0, 1, 0),
        )
        cost_curve_data = {"pcf": pcf.tolist(), "nec": nec.tolist()}
    else:
        cost_curve_data = {"pcf": [], "nec": []}

    # ---- daily / windowed trajectories
    actions_str = [r.engine_action for r in labeled]
    labels_str = [r.label for r in labeled]
    daily_data = []
    if daily:
        for d in daily_metrics(
            (r.timestamp for r in labeled),
            (r.vendor_call for r in labeled),
            actions_str,
            labels_str,
        ):
            daily_data.append({
                "date": d.date, "n": d.n,
                "false_call_rate": d.false_call_rate,
                "escape_rate": d.escape_rate,
                "escalate_rate": d.escalate_rate,
            })

    win = windowed_metrics(vendor_calls, actions_str, labels_str, windowed_window, windowed_step)
    windowed_data = {k: v.tolist() for k, v in win.items()}

    return EvalReport(
        n_total=n_total,
        n_labeled=n_labeled,
        cost_matrix=cm.tolist(),
        engine=engine_summary,
        baseline_vendor_only=bl_summary,
        delta=delta,
        daily=daily_data,
        windowed=windowed_data,
        aurc_engine=engine_aurc,
        cost_curve=cost_curve_data,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scalar_summary(labels: np.ndarray, actions: np.ndarray, expected_cost_value: float) -> dict:
    n = len(labels)
    if n == 0:
        return {}
    is_fc = labels == LABEL_FALSE_CALL
    is_td = labels == LABEL_TRUE_DEFECT
    fc_action_def = (actions == 0) & is_fc
    escape = (actions == 1) & is_td
    escalate = actions == 2
    return {
        "n": int(n),
        "expected_cost": float(expected_cost_value),
        "false_call_rate": float(fc_action_def.sum() / n),
        "escape_rate": float(escape.sum() / n),
        "escalate_rate": float(escalate.sum() / n),
        "n_escapes": int(escape.sum()),
        "n_false_calls": int(fc_action_def.sum()),
        "n_escalations": int(escalate.sum()),
    }
