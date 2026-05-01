"""ROI-level binary classifier: true defect vs false call.

Backbone is timm-based (EfficientNet / ConvNeXt) with a single-logit head.
"""
from __future__ import annotations


def build_classifier(
    backbone: str = "convnext_tiny",
    pretrained: bool = True,
    drop_rate: float = 0.1,
    num_classes: int = 2,
):
    """Return a timm classifier model.

    Default `convnext_tiny` is a good balance for ROI sizes around 224 px.
    For tighter latency budgets, use `efficientnet_b0` or `mobilenetv3_small_100`.
    """
    import timm

    return timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
