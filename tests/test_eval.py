"""Eval pipeline tests.

Karpathy: write the test that proves the metric is computed correctly,
THEN go optimise the model. These are the unit tests for the metrics
themselves — model-level eval is built on top.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

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
from aoi_sentinel.eval.runner import EvalRecord, run_eval
from aoi_sentinel.eval.tracker import (
    FalseCallTracker,
    daily_metrics,
    windowed_metrics,
)


# ---------------------------------------------------------------- expected_cost

def test_expected_cost_zero_when_perfect():
    cm = np.array([[1, 0, 5], [0, 1000, 5]], dtype=np.float32)
    labels = [0, 0, 1, 1]
    actions = [1, 1, 0, 0]   # FALSE_CALL→PASS, TRUE_DEFECT→DEFECT — both correct
    assert expected_cost(labels, actions, cm) == 0.0


def test_expected_cost_catastrophe_dominates():
    cm = np.array([[1, 0, 5], [0, 1000, 5]], dtype=np.float32)
    # 1 escape + 99 perfect → mean ≈ 10
    labels = [1] * 1 + [0] * 99
    actions = [1] * 1 + [1] * 99
    assert abs(expected_cost(labels, actions, cm) - 10.0) < 1e-6


# ----------------------------------------------------------- risk-coverage / AURC

def test_rc_curve_perfect_calibration_gives_zero_aurc():
    # When confidence perfectly ranks errors below corrects, AURC → 0.
    labels = [0, 0, 1, 1]
    preds = [0, 0, 1, 0]   # one error (last)
    confs = [0.9, 0.9, 0.9, 0.1]
    rc = risk_coverage_curve(labels, preds, confs)
    a = aurc(rc)
    # Lower bound of AURC is the area swept while accepting only correct ones first
    assert a < 0.2


def test_rc_curve_uncalibrated_high_aurc():
    # All-wrong classifier on n samples — AURC ≈ (n-1)/n via trapezoidal rule.
    # For n=8 we expect ≈ 0.875.
    labels = [0, 1] * 4
    preds = [1, 0] * 4
    confs = [0.9] * 8
    rc = risk_coverage_curve(labels, preds, confs)
    assert aurc(rc) > 0.85


# ----------------------------------------------------------------- cost curve

def test_cost_curve_shape():
    labels = [0, 0, 1, 1]
    preds = [0, 1, 1, 0]
    pcf, nec = cost_curve(labels, preds, n_points=11)
    assert pcf.shape == (11,)
    assert nec.shape == (11,)
    assert (nec >= 0).all() and (nec <= 1).all()


# ------------------------------------------------------------ FalseCallTracker

def test_false_call_tracker_window_rolls():
    t = FalseCallTracker(window_size=3)
    t.step("DEFECT", "DEFECT", "FALSE_CALL")
    t.step("DEFECT", "DEFECT", "FALSE_CALL")
    t.step("DEFECT", "DEFECT", "TRUE_DEFECT")
    assert abs(t.false_call_rate - 2/3) < 1e-9
    t.step("DEFECT", "PASS", "TRUE_DEFECT")  # escape
    assert len(t) == 3
    assert t.escape_rate == 1/3


# ------------------------------------------------------------- daily / windowed

def test_daily_metrics_groups_by_date():
    ts = [
        datetime(2026, 4, 28, 10, 0, tzinfo=timezone.utc),
        datetime(2026, 4, 28, 11, 0, tzinfo=timezone.utc),
        datetime(2026, 4, 29, 9, 0, tzinfo=timezone.utc),
    ]
    vc = ["DEFECT", "DEFECT", "DEFECT"]
    ac = ["DEFECT", "DEFECT", "PASS"]
    lb = ["FALSE_CALL", "TRUE_DEFECT", "TRUE_DEFECT"]
    days = daily_metrics(ts, vc, ac, lb)
    assert len(days) == 2
    assert days[0].date == "2026-04-28"
    assert days[1].date == "2026-04-29"
    assert days[1].escape_rate == 1.0


def test_windowed_metrics_skips_when_short():
    out = windowed_metrics(
        ["DEFECT"] * 5, ["DEFECT"] * 5, ["FALSE_CALL"] * 5,
        window_size=10, step=1,
    )
    assert out["idx"].size == 0


# ------------------------------------------------------------------ baselines

def test_vendor_only_baseline_pass_through():
    res = vendor_only_baseline(["DEFECT", "PASS", "UNKNOWN"])
    assert list(res.actions) == ["DEFECT", "PASS", "ESCALATE"]


def test_selective_threshold_baseline_three_actions():
    res = selective_threshold_baseline([0.9, 0.5, 0.1], threshold=0.5, abstain_band=0.1)
    assert list(res.actions) == ["DEFECT", "ESCALATE", "PASS"]


# ----------------------------------------------------------------- run_eval

def test_run_eval_end_to_end():
    ts0 = datetime(2026, 4, 29, 10, 0, tzinfo=timezone.utc)
    records = [
        EvalRecord(ts0, "DEFECT", "DEFECT", 0.9, "TRUE_DEFECT"),
        EvalRecord(ts0, "DEFECT", "PASS",   0.8, "FALSE_CALL"),
        EvalRecord(ts0, "DEFECT", "PASS",   0.6, "FALSE_CALL"),
        EvalRecord(ts0, "DEFECT", "ESCALATE", 0.4, "TRUE_DEFECT"),
    ]
    rep = run_eval(records)
    assert rep.n_total == 4
    assert rep.n_labeled == 4
    assert rep.engine["n_escapes"] == 0
    assert rep.engine["n_false_calls"] == 0     # we PASS'd both FC correctly
    assert rep.baseline_vendor_only["n_false_calls"] == 2
    assert rep.delta["false_call_drop_pp"] > 0   # we beat the baseline on FC


def test_run_eval_handles_empty():
    rep = run_eval([])
    assert rep.n_total == 0
    assert rep.n_labeled == 0
