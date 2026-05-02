"""Cost-sensitive binary classifier — trainable, deployable, edge-friendly.

This is the production v0 model. It wraps `LightweightEncoder` (MobileNetV3 /
ConvNeXt-Pico) with a 2-class head and supports a configurable reject
threshold so the inferencer emits {DEFECT, PASS, ESCALATE} per Chow's rule.

The Mamba RL stack lives in `models/policy/` and `models/vmamba/`. That's
the research arm. *This* module is what actually ships first.
"""
from aoi_sentinel.models.classifier.model import LightweightClassifier
from aoi_sentinel.models.classifier.infer import Inferencer
from aoi_sentinel.models.classifier.train import train_classifier
from aoi_sentinel.models.classifier.dataset import LabelDataset, build_dataset

__all__ = [
    "LightweightClassifier",
    "Inferencer",
    "train_classifier",
    "LabelDataset",
    "build_dataset",
]
