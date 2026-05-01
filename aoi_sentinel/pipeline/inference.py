"""Top-level inference pipeline.

Pipeline:
    Saki call (defect ROIs)
        → 2D false-call classifier            (Phase 1)
        → 3D height-map analyzer (if 3D data) (Phase 3)
        → fused verdict
        → RAG cause reasoner (optional)        (Phase 4)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FinalVerdict:
    is_true_defect: bool
    confidence: float
    defect_type: str | None
    cause_hypothesis: str | None = None
    similar_history: list[str] | None = None


def run_pipeline(*args, **kwargs) -> FinalVerdict:  # pragma: no cover
    raise NotImplementedError("Phase 1 classifier ships first; pipeline wires later.")
