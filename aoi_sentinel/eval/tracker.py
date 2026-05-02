"""Time-resolved trackers.

Scalar metrics tell you "are we good on average". Trajectories tell you
**whether we are still improving** — which is the only signal that matters in
NPI mode. A model that has plateaued must be flagged so the trainer knows
to retrain on more recent data.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Streaming tracker  — the runtime keeps one of these per line
# ---------------------------------------------------------------------------


@dataclass
class FalseCallTracker:
    """Rolling-window tracker for live monitoring.

    `step()` is called once per labelled decision. The tracker maintains a
    fixed-size sliding window of the most recent N decisions and exposes
    `false_call_rate`, `escape_rate`, `escalate_rate` over that window.

    For a stronger trend signal use a longer window (e.g. 5_000) — it lags
    but reduces variance.
    """

    window_size: int = 1_000
    _window: deque = field(default_factory=deque)

    # Running counts on the windowed buffer
    _fc: int = 0       # vendor said DEFECT, label says FALSE_CALL
    _esc: int = 0      # we said ESCALATE
    _escape: int = 0   # vendor would have shipped a true defect (label=TRUE_DEFECT, action=PASS)

    def step(self, vendor_call: str, action: str, label: str) -> None:
        """One labelled decision."""
        is_fc = vendor_call == "DEFECT" and label == "FALSE_CALL"
        is_escalate = action == "ESCALATE"
        is_escape = action == "PASS" and label == "TRUE_DEFECT"
        rec = (int(is_fc), int(is_escalate), int(is_escape))
        if len(self._window) == self.window_size:
            old = self._window.popleft()
            self._fc -= old[0]; self._esc -= old[1]; self._escape -= old[2]
        self._window.append(rec)
        self._fc += rec[0]; self._esc += rec[1]; self._escape += rec[2]

    def __len__(self) -> int:
        return len(self._window)

    @property
    def false_call_rate(self) -> float:
        return self._fc / len(self._window) if self._window else 0.0

    @property
    def escape_rate(self) -> float:
        return self._escape / len(self._window) if self._window else 0.0

    @property
    def escalate_rate(self) -> float:
        return self._esc / len(self._window) if self._window else 0.0


# ---------------------------------------------------------------------------
# Daily / windowed aggregates  — for offline reports and dashboard plots
# ---------------------------------------------------------------------------


@dataclass
class DailyMetrics:
    date: str          # YYYY-MM-DD
    n: int
    false_call_rate: float
    escape_rate: float
    escalate_rate: float


def daily_metrics(
    timestamps: Iterable[datetime],
    vendor_calls: Iterable[str],
    actions: Iterable[str],
    labels: Iterable[str],
) -> list[DailyMetrics]:
    """Group by UTC date; return a sorted list of `DailyMetrics`."""
    buckets: dict[str, list[tuple[str, str, str]]] = {}
    for ts, vc, ac, lb in zip(timestamps, vendor_calls, actions, labels):
        key = (ts.astimezone(timezone.utc) if ts.tzinfo else ts).date().isoformat()
        buckets.setdefault(key, []).append((vc, ac, lb))

    out: list[DailyMetrics] = []
    for date in sorted(buckets):
        rows = buckets[date]
        n = len(rows)
        fc = sum(1 for vc, ac, lb in rows if vc == "DEFECT" and lb == "FALSE_CALL")
        esc = sum(1 for vc, ac, lb in rows if ac == "ESCALATE")
        escape = sum(1 for vc, ac, lb in rows if ac == "PASS" and lb == "TRUE_DEFECT")
        out.append(
            DailyMetrics(
                date=date,
                n=n,
                false_call_rate=fc / n,
                escape_rate=escape / n,
                escalate_rate=esc / n,
            )
        )
    return out


def windowed_metrics(
    vendor_calls: Iterable[str],
    actions: Iterable[str],
    labels: Iterable[str],
    window_size: int = 1_000,
    step: int = 100,
) -> dict[str, np.ndarray]:
    """Sliding-window aggregates over a fixed-size queue.

    Returns dict with arrays:
        idx              — right edge of each window (0-indexed step count)
        false_call_rate
        escape_rate
        escalate_rate
    """
    vcs = list(vendor_calls); acs = list(actions); lbs = list(labels)
    n = len(vcs)
    if not (len(acs) == len(lbs) == n):
        raise ValueError("length mismatch")
    if n < window_size:
        return {
            "idx": np.array([]),
            "false_call_rate": np.array([]),
            "escape_rate": np.array([]),
            "escalate_rate": np.array([]),
        }

    idxs, fcs, escs, escapes = [], [], [], []
    for end in range(window_size, n + 1, step):
        start = end - window_size
        win_v = vcs[start:end]; win_a = acs[start:end]; win_l = lbs[start:end]
        fcs.append(sum(1 for v, l in zip(win_v, win_l) if v == "DEFECT" and l == "FALSE_CALL") / window_size)
        escs.append(sum(1 for a in win_a if a == "ESCALATE") / window_size)
        escapes.append(sum(1 for a, l in zip(win_a, win_l) if a == "PASS" and l == "TRUE_DEFECT") / window_size)
        idxs.append(end)

    return {
        "idx": np.asarray(idxs),
        "false_call_rate": np.asarray(fcs),
        "escape_rate": np.asarray(escapes),
        "escalate_rate": np.asarray(escs),
    }
