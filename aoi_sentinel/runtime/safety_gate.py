"""Safety gate for model promotion.

A candidate model is promoted only if it satisfies, on a held-out replay
of recent labelled boards:

  1. zero escapes  (true defect → PASS) — non-negotiable
  2. false-call reduction strictly better than the current model
  3. ESCALATE rate not catastrophically higher than the current model
     (otherwise we just trade false calls for operator load)

The hold-out is the most recent K labelled boards that the candidate did
NOT train on. K is chosen large enough that one missed escape is statistically
significant given the production defect rate.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GateConfig:
    min_holdout_size: int = 5_000
    max_escapes_allowed: int = 0
    min_fc_reduction_pp: float = 0.005      # 0.5 percentage points improvement
    max_escalate_increase_pp: float = 0.05  # ESCALATE may rise at most 5 pp


@dataclass
class CandidateScore:
    n: int
    escapes: int
    fc_rate: float
    escalate_rate: float


@dataclass
class GateResult:
    passed: bool
    reasons: list[str]
    candidate: CandidateScore
    incumbent: CandidateScore


def evaluate(
    candidate: CandidateScore,
    incumbent: CandidateScore,
    cfg: GateConfig | None = None,
) -> GateResult:
    cfg = cfg or GateConfig()
    reasons: list[str] = []

    if candidate.n < cfg.min_holdout_size:
        reasons.append(f"holdout too small ({candidate.n} < {cfg.min_holdout_size})")

    if candidate.escapes > cfg.max_escapes_allowed:
        reasons.append(
            f"candidate escaped {candidate.escapes} true defects "
            f"(max {cfg.max_escapes_allowed})"
        )

    fc_delta = incumbent.fc_rate - candidate.fc_rate
    if fc_delta < cfg.min_fc_reduction_pp:
        reasons.append(
            f"insufficient false-call reduction "
            f"({fc_delta * 100:.2f} pp < {cfg.min_fc_reduction_pp * 100:.2f} pp)"
        )

    esc_delta = candidate.escalate_rate - incumbent.escalate_rate
    if esc_delta > cfg.max_escalate_increase_pp:
        reasons.append(
            f"escalate rate increased too much "
            f"({esc_delta * 100:.2f} pp > {cfg.max_escalate_increase_pp * 100:.2f} pp)"
        )

    return GateResult(
        passed=not reasons,
        reasons=reasons,
        candidate=candidate,
        incumbent=incumbent,
    )
