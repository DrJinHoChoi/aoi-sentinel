"""Simulated operator. In the real line, ESCALATE → human re-inspects → label arrives.

Here we have a benchmark dataset with ground-truth labels, so the oracle simply
reveals the label on demand. We separate it from the env to keep the abstraction
clean for when we plug in a real operator queue.
"""
from __future__ import annotations

from typing import Protocol


class LabelOracle(Protocol):
    """Reveals ground-truth label for a given sample id."""

    def reveal(self, sample_id: int) -> int:
        ...


class GroundTruthOracle:
    """Returns the true label from an in-memory array."""

    def __init__(self, labels):
        self._labels = labels

    def reveal(self, sample_id: int) -> int:
        return int(self._labels[sample_id])
