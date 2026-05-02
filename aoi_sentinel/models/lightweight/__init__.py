"""Lightweight image encoders for edge inference.

The Mamba stack lives on the trainer (DGX Spark). Edge boxes (Jetson Orin Nano,
or even a 2019 Jetson Nano for POC demos) need a much smaller model that:

  - exports cleanly to TensorRT / ONNX
  - hits ~10-30 FPS at 224² on Maxwell/Ampere edge GPUs
  - keeps the same (B, embed_dim) contract as the heavyweight encoder

We pick MobileNetV3-Small as the default and ConvNeXt-Pico as the slightly
heavier "edge but Orin NX or better" option. Both are timm-registered and
quantize well.
"""
from aoi_sentinel.models.lightweight.encoder import LightweightEncoder, build_lightweight_encoder

__all__ = ["LightweightEncoder", "build_lightweight_encoder"]
