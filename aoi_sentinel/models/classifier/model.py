"""Cost-sensitive binary classifier — `LightweightEncoder` + linear head.

Output: (B, 2) logits over {0=FALSE_CALL, 1=TRUE_DEFECT}. The inferencer
applies softmax + a configurable threshold to derive the 3-action
{DEFECT, PASS, ESCALATE} decision via Chow's rule.

Why a binary head and not a 3-class softmax over actions:
  - "ESCALATE" is a *decision rule*, not a class. The data has only two
    possible labels (true defect vs false call); there's no third label to
    learn from.
  - Calibration matters for the reject rule, and the binary softmax
    calibrates more cleanly than a 3-way softmax with an "unsure" head.
  - Same backbone can be re-used by the offline RL fine-tune in
    `models/policy/` without retraining from scratch.
"""
from __future__ import annotations


class LightweightClassifier:
    """Lazy-imported torch wrapper. Module-level import never touches torch."""

    def __new__(cls, *args, **kwargs):
        # Only realise the torch.nn.Module on first instantiation.
        from aoi_sentinel.models.classifier._impl import _LightweightClassifierImpl
        return _LightweightClassifierImpl(*args, **kwargs)
