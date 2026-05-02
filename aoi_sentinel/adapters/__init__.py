"""Vendor adapters — one module per AOI maker."""
from aoi_sentinel.adapters.base import (
    CommonInspection,
    ComponentInspection,
    EngineAction,
    Verdict,
    VendorAdapter,
    VendorVerdict,
    available_adapters,
    make_adapter,
    register,
)

# Trigger registration side-effects on these submodules.
from aoi_sentinel.adapters import generic_csv  # noqa: F401
from aoi_sentinel.adapters import koh_young    # noqa: F401
from aoi_sentinel.adapters import saki         # noqa: F401

__all__ = [
    "CommonInspection",
    "ComponentInspection",
    "EngineAction",
    "Verdict",
    "VendorAdapter",
    "VendorVerdict",
    "available_adapters",
    "make_adapter",
    "register",
]
