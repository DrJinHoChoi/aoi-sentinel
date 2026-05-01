"""Public benchmark loaders. Each returns (images, labels, saki_calls) for the NpiEnv."""
from aoi_sentinel.data.benchmarks.deeppcb import load_deeppcb
from aoi_sentinel.data.benchmarks.soldef import load_soldef
from aoi_sentinel.data.benchmarks.visa import load_visa

__all__ = ["load_visa", "load_deeppcb", "load_soldef"]
