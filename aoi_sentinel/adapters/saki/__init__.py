"""Saki AOI adapter.

Saki BF-3Di / 3Di-LU / SD-Series typically dump per-board:
  - 2D top-view colour image(s)
  - 3D height map / depth (TIFF or proprietary .ASC)
  - Inspection result XML (per-board) with component-level judgements

This adapter watches the Saki output share, parses each board's XML, and
yields a `CommonInspection`. Implementation completes when we have a real
Saki sample dump — the schema is well-defined upstream.
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Iterator

from aoi_sentinel.adapters.base import (
    CommonInspection,
    ComponentInspection,
    Verdict,
    register,
)


@register("saki")
class SakiAdapter:
    name: str

    def __init__(self, poll_interval_s: float = 2.0) -> None:
        self.poll_interval_s = poll_interval_s
        self._seen: set[str] = set()

    # ------------------------------------------------------------------ watch

    def watch(self, source: str | Path) -> Iterator[CommonInspection]:
        """Watch a Saki result share. Each new XML triggers one CommonInspection."""
        root = Path(source)
        if not root.exists():
            raise FileNotFoundError(root)

        while True:
            for xml_path in sorted(root.rglob("*.xml")):
                if str(xml_path) in self._seen:
                    continue
                try:
                    insp = self._parse_board_xml(xml_path)
                except NotImplementedError:
                    raise
                except Exception as e:  # noqa: BLE001
                    print(f"[saki] skip malformed {xml_path}: {e}")
                    self._seen.add(str(xml_path))
                    continue
                self._seen.add(str(xml_path))
                yield insp
            time.sleep(self.poll_interval_s)

    # ------------------------------------------------------------------ parse

    def _parse_board_xml(self, xml_path: Path) -> CommonInspection:
        """Parse one Saki board XML.

        TODO: fill in once a real sample arrives. Skeleton below shows the
        intended call-site contract — real fields and namespaces vary by
        Saki software version (PowerScout vs. PowerView) and recipe.
        """
        # Pseudocode skeleton — replace when first real sample is in hand.
        #
        #   root = lxml.etree.parse(xml_path).getroot()
        #   board_id  = root.attrib["BoardID"]
        #   timestamp = datetime.fromisoformat(root.attrib["Time"])
        #   line_id   = root.attrib.get("Line")
        #   lot       = root.attrib.get("Lot")
        #   components = []
        #   for c in root.iter("Component"):
        #       img = cv2.imread(str(xml_path.parent / c.attrib["ImagePath"]))
        #       components.append(ComponentInspection(
        #           ref_des = c.attrib["RefDes"],
        #           bbox_xyxy = (int(c.attrib["X1"]), int(c.attrib["Y1"]),
        #                        int(c.attrib["X2"]), int(c.attrib["Y2"])),
        #           image_2d = img,
        #           vendor_call = "DEFECT" if c.attrib["Result"] == "NG" else "PASS",
        #           vendor_defect_type = c.attrib.get("DefectType"),
        #       ))
        #   return CommonInspection(board_id, timestamp, "saki",
        #                            line_id, lot, components, str(xml_path))
        raise NotImplementedError(
            "Saki XML parser pending — drop a real one-board sample dump and we "
            "fill this in. Schema differs across Saki software versions."
        )

    # ---------------------------------------------------------------- verdict

    def push_verdict(self, board_id: str, verdicts: list[Verdict]) -> None:
        """Saki MES integration TBD. Phase 4 of vendor SDK work."""
        # Saki Core Studio supports an external-decision API on some lines.
        # Until we have the API spec, we log + write a sidecar JSON next to the
        # original XML so MES can pick it up via existing share watch.
        print(f"[saki] verdicts board={board_id} ({len(verdicts)}) — sidecar TODO")
