"""Saki AOI inspection dump parser.

Saki BF-3Di / 3Di-LU style outputs typically include, per board:
  - 2D top-view color image(s)
  - 3D height map / depth (TIFF or proprietary)
  - Inspection result XML/CSV with per-component judgements

This module is a placeholder until the actual Saki export format is confirmed.
Once we have a sample dump, fill in `_parse_result_xml` and `_load_height_map`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass
class ComponentInspection:
    """One component (reference designator) on one board, one inspection."""

    board_id: str
    ref_des: str                 # e.g. "C12", "U3"
    bbox_xyxy: tuple[int, int, int, int]
    saki_verdict: str            # raw Saki call, e.g. "DEFECT", "PASS"
    saki_defect_type: str | None  # e.g. "MISSING", "REVERSED", "SHORT", "TOMBSTONE"
    operator_verdict: str | None  # ground truth from rework/reinspection: "TRUE_DEFECT" | "FALSE_CALL" | None
    image_2d_path: str
    height_map_path: str | None
    extra: dict = field(default_factory=dict)


@dataclass
class BoardInspection:
    board_id: str
    lot: str | None
    timestamp: str | None
    components: list[ComponentInspection] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan_saki_dump(root: str | Path, out_parquet: str | Path) -> int:
    """Walk a Saki dump root and write an indexed parquet of all components.

    The exact directory layout depends on Saki configuration. This function
    is the entry point — implementation deferred until a real sample arrives.

    Returns the number of indexed components.
    """
    raise NotImplementedError(
        "scan_saki_dump: provide a Saki sample dump (XML + image layout) to implement"
    )


def _parse_result_xml(xml_path: Path) -> BoardInspection:
    """Parse a Saki inspection result XML into a BoardInspection."""
    raise NotImplementedError


def _load_height_map(path: Path):
    """Load a Saki 3D height map as a 2D numpy array (mm or microns)."""
    raise NotImplementedError
