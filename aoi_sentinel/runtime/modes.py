"""Operating-mode state machine.

Three modes, governed by a single rule: never regress on escape rate.

    SHADOW       — engine watches, displays its decision next to the vendor's,
                   but the operator decides everything. Pure label collection.
                   Default for the first N boards on every new product.

    ASSIST       — engine + operator agree → auto. Disagreement → operator decides.
                   The "human-in-the-loop" steady state.

    AUTONOMOUS   — engine decides, MES gets verdict directly. Operator monitors
                   KPI dashboard; spot-checks via sampling.

Transitions are gated by `safety_gate`. Promotion to a more autonomous mode
requires a sustained window where:
    - escape_rate (true defect → PASS) == 0
    - false_call_rate is below a target
    - sample size meets a minimum
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Mode(str, Enum):
    SHADOW = "SHADOW"
    ASSIST = "ASSIST"
    AUTONOMOUS = "AUTONOMOUS"


@dataclass
class ModeGate:
    """Promotion criteria. Set per product or per line in config."""

    min_boards_for_assist: int = 1_000
    min_boards_for_autonomous: int = 50_000

    max_escape_for_assist: int = 0           # zero misses required
    max_escape_for_autonomous: int = 0       # zero misses required

    max_fc_rate_for_autonomous: float = 0.005  # < 0.5% false calls

    consecutive_days_required: int = 30      # for AUTONOMOUS only


@dataclass
class ModeStats:
    """Rolling stats fed by the runtime."""

    boards_seen: int
    escapes_in_window: int
    false_calls_in_window: int
    components_in_window: int
    consecutive_clean_days: int

    @property
    def fc_rate(self) -> float:
        if not self.components_in_window:
            return 0.0
        return self.false_calls_in_window / self.components_in_window


def next_mode(current: Mode, stats: ModeStats, gate: ModeGate) -> Mode:
    """Determine the next allowed mode. Never auto-demotes here — demotion is
    handled by the safety gate when an escape happens."""
    if current is Mode.SHADOW:
        if (
            stats.boards_seen >= gate.min_boards_for_assist
            and stats.escapes_in_window <= gate.max_escape_for_assist
        ):
            return Mode.ASSIST
        return Mode.SHADOW

    if current is Mode.ASSIST:
        if (
            stats.boards_seen >= gate.min_boards_for_autonomous
            and stats.escapes_in_window <= gate.max_escape_for_autonomous
            and stats.fc_rate <= gate.max_fc_rate_for_autonomous
            and stats.consecutive_clean_days >= gate.consecutive_days_required
        ):
            return Mode.AUTONOMOUS
        return Mode.ASSIST

    return Mode.AUTONOMOUS


def demote_on_escape(current: Mode) -> Mode:
    """Single escape event → drop a level. Never silently stay in higher mode."""
    if current is Mode.AUTONOMOUS:
        return Mode.ASSIST
    if current is Mode.ASSIST:
        return Mode.SHADOW
    return Mode.SHADOW
