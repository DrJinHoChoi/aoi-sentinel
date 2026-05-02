"""Runtime smoke tests — modes, safety gate, label queue, model registry."""
from __future__ import annotations

from datetime import datetime

import pytest

from aoi_sentinel.runtime.label_queue import LabelQueue, LabelRecord
from aoi_sentinel.runtime.model_registry import ModelRegistry
from aoi_sentinel.runtime.modes import (
    Mode,
    ModeGate,
    ModeStats,
    demote_on_escape,
    next_mode,
)
from aoi_sentinel.runtime.safety_gate import (
    CandidateScore,
    GateConfig,
    evaluate,
)


# ----------------------------------------------------------------- modes

def test_shadow_holds_until_threshold():
    stats = ModeStats(boards_seen=500, escapes_in_window=0, false_calls_in_window=10,
                     components_in_window=1000, consecutive_clean_days=5)
    assert next_mode(Mode.SHADOW, stats, ModeGate()) is Mode.SHADOW


def test_shadow_promotes_to_assist_after_threshold():
    stats = ModeStats(boards_seen=2000, escapes_in_window=0, false_calls_in_window=20,
                     components_in_window=2000, consecutive_clean_days=10)
    assert next_mode(Mode.SHADOW, stats, ModeGate()) is Mode.ASSIST


def test_assist_holds_without_clean_days():
    stats = ModeStats(boards_seen=60_000, escapes_in_window=0, false_calls_in_window=10,
                     components_in_window=10_000, consecutive_clean_days=5)
    assert next_mode(Mode.ASSIST, stats, ModeGate()) is Mode.ASSIST


def test_assist_promotes_to_autonomous_when_all_pass():
    stats = ModeStats(boards_seen=60_000, escapes_in_window=0, false_calls_in_window=10,
                     components_in_window=10_000, consecutive_clean_days=30)
    assert next_mode(Mode.ASSIST, stats, ModeGate()) is Mode.AUTONOMOUS


def test_escape_demotes_one_step():
    assert demote_on_escape(Mode.AUTONOMOUS) is Mode.ASSIST
    assert demote_on_escape(Mode.ASSIST) is Mode.SHADOW
    assert demote_on_escape(Mode.SHADOW) is Mode.SHADOW


# ------------------------------------------------------------ safety gate

def test_gate_passes_when_better():
    cand = CandidateScore(n=10_000, escapes=0, fc_rate=0.04, escalate_rate=0.05)
    inc = CandidateScore(n=10_000, escapes=0, fc_rate=0.07, escalate_rate=0.05)
    r = evaluate(cand, inc, GateConfig())
    assert r.passed, r.reasons


def test_gate_rejects_on_escape():
    cand = CandidateScore(n=10_000, escapes=1, fc_rate=0.01, escalate_rate=0.05)
    inc = CandidateScore(n=10_000, escapes=0, fc_rate=0.07, escalate_rate=0.05)
    r = evaluate(cand, inc, GateConfig())
    assert not r.passed
    assert any("escaped" in s for s in r.reasons)


def test_gate_rejects_on_fc_regression():
    cand = CandidateScore(n=10_000, escapes=0, fc_rate=0.07, escalate_rate=0.05)
    inc = CandidateScore(n=10_000, escapes=0, fc_rate=0.07, escalate_rate=0.05)
    r = evaluate(cand, inc, GateConfig())
    assert not r.passed


# ------------------------------------------------------------- label queue

def test_label_queue_round_trip(tmp_path):
    q = LabelQueue(tmp_path / "labels.db")
    rec = LabelRecord(
        board_id="B-1", ref_des="C12", vendor="generic_csv", line_id="L1",
        timestamp=datetime.utcnow(),
        image_path="/tmp/x.jpg", height_map_path=None,
        vendor_call="DEFECT", vendor_defect_type="MISSING",
        engine_action="ESCALATE", engine_confidence=0.6,
        operator_label="FALSE_CALL", operator_id="op-1",
        model_version="v0",
    )
    q.append(rec)
    q.append(rec)
    assert q.count() == 2
    rows = list(q.stream_since(0))
    assert len(rows) == 2
    assert rows[0].board_id == "B-1"


# ------------------------------------------------------------ model registry

def test_model_registry_stage_and_promote(tmp_path):
    reg = ModelRegistry(tmp_path / "registry")
    weights = tmp_path / "w.pt"; weights.write_bytes(b"weights")
    config = tmp_path / "c.yaml"; config.write_text("dummy: true")
    h = reg.stage("v1", weights, config, metadata={"note": "first"})
    assert h.metadata["promoted"] is False
    assert reg.current() is None  # not promoted yet
    reg.promote("v1")
    cur = reg.current()
    assert cur is not None
    assert cur.version == "v1"
    assert cur.metadata["promoted"] is True


def test_model_registry_rollback(tmp_path):
    reg = ModelRegistry(tmp_path / "registry")
    for v in ("v1", "v2"):
        w = tmp_path / f"{v}.pt"; w.write_bytes(b"w")
        c = tmp_path / f"{v}.yaml"; c.write_text("d: 1")
        reg.stage(v, w, c)
        reg.promote(v)
    assert reg.current().version == "v2"
    reg.rollback("v1")
    assert reg.current().version == "v1"
