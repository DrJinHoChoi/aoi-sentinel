"""Cost matrix for the NPI MDP."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Action ids (also used in NpiEnv.action_space)
ACTION_DEFECT = 0
ACTION_PASS = 1
ACTION_ESCALATE = 2

# Ground-truth labels
LABEL_FALSE_CALL = 0   # Saki flagged but the part is fine
LABEL_TRUE_DEFECT = 1  # Saki flagged and the part really is defective


@dataclass(frozen=True)
class CostMatrix:
    """Cost incurred per (true label, action) pair.

    Default assumes automotive electronics — an escape is roughly 1000× the
    cost of a false call, with operator review somewhere in between.
    """

    c_escape: float = 1000.0          # y=TRUE_DEFECT, a=PASS — the catastrophe
    c_false_call: float = 1.0         # y=FALSE_CALL,  a=DEFECT — needless rework
    c_operator: float = 5.0           # any a=ESCALATE — operator labor cost
    c_correct_defect: float = 0.0     # y=TRUE_DEFECT, a=DEFECT — Saki was right
    c_correct_pass: float = 0.0       # y=FALSE_CALL,  a=PASS  — we correctly cleared

    def matrix(self) -> np.ndarray:
        """2x3 matrix indexed [label, action]."""
        m = np.zeros((2, 3), dtype=np.float32)
        m[LABEL_FALSE_CALL, ACTION_DEFECT] = self.c_false_call
        m[LABEL_FALSE_CALL, ACTION_PASS] = self.c_correct_pass
        m[LABEL_FALSE_CALL, ACTION_ESCALATE] = self.c_operator
        m[LABEL_TRUE_DEFECT, ACTION_DEFECT] = self.c_correct_defect
        m[LABEL_TRUE_DEFECT, ACTION_PASS] = self.c_escape
        m[LABEL_TRUE_DEFECT, ACTION_ESCALATE] = self.c_operator
        return m

    def is_escape(self, label: int, action: int) -> bool:
        return label == LABEL_TRUE_DEFECT and action == ACTION_PASS


def default_cost_matrix() -> CostMatrix:
    return CostMatrix()
