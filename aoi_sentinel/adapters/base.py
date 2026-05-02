"""Vendor-agnostic AOI adapter SDK.

Every supported AOI maker (Saki, Koh Young, Mycronic, TRI, Mirtec, Omron, ...)
ships a different output format. This module defines the **only** contract the
core engine sees: vendors live behind a thin adapter that normalises native
output to the `CommonInspection` schema and pushes verdicts back.

Adding a new vendor = implement `VendorAdapter`, ship one set of unit tests,
done. No core-engine changes required.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Literal, Protocol, runtime_checkable

import numpy as np

# ---------------------------------------------------------------------------
# Common schema
# ---------------------------------------------------------------------------

VendorVerdict = Literal["PASS", "DEFECT", "UNKNOWN"]
DefectType = str  # vendor-defined string; we don't over-constrain here


@dataclass
class ComponentInspection:
    """One component (reference designator) on one board, one inspection."""

    ref_des: str                       # e.g. "C12", "U3"
    bbox_xyxy: tuple[int, int, int, int]
    image_2d: np.ndarray               # HWC uint8, vendor-cropped to ROI region
    height_map: np.ndarray | None = None  # 2D float32 (mm or microns) if 3D AOI
    vendor_call: VendorVerdict = "DEFECT"
    vendor_defect_type: DefectType | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class CommonInspection:
    """One board's worth of inspection result, normalised."""

    board_id: str                      # vendor barcode/serial, must be unique
    timestamp: datetime
    vendor: str                        # adapter name, e.g. "saki", "koh_young"
    line_id: str | None                # optional line/station identifier
    lot: str | None                    # optional production lot
    components: list[ComponentInspection]
    raw_payload_path: str | None = None  # for traceability, original vendor file

    def __len__(self) -> int:
        return len(self.components)


# ---------------------------------------------------------------------------
# Verdict (engine → adapter → vendor system / MES)
# ---------------------------------------------------------------------------

EngineAction = Literal["DEFECT", "PASS", "ESCALATE"]


@dataclass
class Verdict:
    """Our decision for a single component."""

    ref_des: str
    action: EngineAction
    confidence: float                   # [0, 1]
    rationale: str | None = None        # short human-readable explanation
    model_version: str = ""


# ---------------------------------------------------------------------------
# Adapter protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class VendorAdapter(Protocol):
    """Implement this for each AOI vendor.

    Lifecycle:
        1. construct(config)          — read connection params, set up watchers
        2. for evt in adapter.watch():  — yield events as new boards arrive
        3. push_verdict(...)            — optionally relay engine decisions back

    Adapters must be **stateless across boards** as far as the engine sees —
    any per-line drift modelling lives in the engine's sequence encoder, not here.
    """

    name: str

    def watch(self, source: str | Path) -> Iterator[CommonInspection]:
        """Yield a `CommonInspection` for every new board the AOI completes.

        `source` is a vendor-specific reference: a watched folder path, a TCP
        endpoint, an OPC-UA node, etc. Adapters interpret it themselves.
        Implementations should be resilient to malformed or partial files.
        """
        ...

    def push_verdict(
        self,
        board_id: str,
        verdicts: list[Verdict],
    ) -> None:
        """Send engine verdicts back to the vendor's system (optional).

        Used for MES integration in AUTONOMOUS mode. Adapters that have no
        return channel may implement this as a no-op + log.
        """
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type] = {}


def register(name: str):
    """Decorator: register a vendor adapter under a stable name."""

    def deco(cls):
        if name in _REGISTRY:
            raise ValueError(f"adapter '{name}' already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def available_adapters() -> list[str]:
    return sorted(_REGISTRY.keys())


def make_adapter(name: str, **kwargs) -> VendorAdapter:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown adapter '{name}'. available: {available_adapters()}. "
            f"new vendor? implement VendorAdapter and decorate with @register('{name}')."
        )
    return _REGISTRY[name](**kwargs)
