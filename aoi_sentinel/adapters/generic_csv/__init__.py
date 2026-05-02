"""Generic CSV folder-watch adapter.

The fallback for any AOI maker that exports a result folder with images and a
CSV index. Useful for:
  - smoke-testing the engine on benchmark data without writing a real adapter
  - quick proof-of-concept on a vendor whose dedicated adapter doesn't exist yet
  - operator-supplied corner cases dropped into a folder

Expected CSV columns (header required):
    board_id,timestamp,line_id,lot,ref_des,
    bbox_x1,bbox_y1,bbox_x2,bbox_y2,
    image_path,height_map_path,vendor_call,vendor_defect_type
"""
from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from aoi_sentinel.adapters.base import (
    CommonInspection,
    ComponentInspection,
    Verdict,
    register,
)


@register("generic_csv")
class GenericCsvAdapter:
    name: str  # set by @register

    def __init__(self, poll_interval_s: float = 2.0) -> None:
        self.poll_interval_s = poll_interval_s
        self._seen: set[str] = set()

    # ------------------------------------------------------------------ watch

    def watch(self, source: str | Path) -> Iterator[CommonInspection]:
        root = Path(source)
        if not root.exists():
            raise FileNotFoundError(root)

        while True:
            for csv_path in sorted(root.glob("*.csv")):
                if csv_path.name in self._seen:
                    continue
                try:
                    insp = self._parse_csv(csv_path, root)
                except Exception as e:  # noqa: BLE001 — adapter must be resilient
                    print(f"[generic_csv] skip malformed {csv_path}: {e}")
                    self._seen.add(csv_path.name)
                    continue
                self._seen.add(csv_path.name)
                yield insp
            time.sleep(self.poll_interval_s)

    def _parse_csv(self, csv_path: Path, root: Path) -> CommonInspection:
        components: list[ComponentInspection] = []
        board_id = ""
        timestamp = datetime.utcnow()
        line_id: str | None = None
        lot: str | None = None

        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                board_id = row["board_id"]
                timestamp = datetime.fromisoformat(row["timestamp"])
                line_id = row.get("line_id") or None
                lot = row.get("lot") or None

                img_path = root / row["image_path"]
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img is None:
                    raise FileNotFoundError(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                height_map = None
                hp = row.get("height_map_path")
                if hp:
                    hp_path = root / hp
                    height_map = np.load(hp_path) if hp_path.suffix == ".npy" else None

                components.append(
                    ComponentInspection(
                        ref_des=row["ref_des"],
                        bbox_xyxy=(
                            int(row["bbox_x1"]),
                            int(row["bbox_y1"]),
                            int(row["bbox_x2"]),
                            int(row["bbox_y2"]),
                        ),
                        image_2d=img,
                        height_map=height_map,
                        vendor_call=row.get("vendor_call", "DEFECT"),
                        vendor_defect_type=row.get("vendor_defect_type") or None,
                    )
                )

        return CommonInspection(
            board_id=board_id,
            timestamp=timestamp,
            vendor="generic_csv",
            line_id=line_id,
            lot=lot,
            components=components,
            raw_payload_path=str(csv_path),
        )

    # ---------------------------------------------------------------- verdict

    def push_verdict(self, board_id: str, verdicts: list[Verdict]) -> None:
        """No back-channel — generic_csv is a one-way adapter. We log instead."""
        n_pass = sum(1 for v in verdicts if v.action == "PASS")
        n_def = sum(1 for v in verdicts if v.action == "DEFECT")
        n_esc = sum(1 for v in verdicts if v.action == "ESCALATE")
        print(
            f"[generic_csv] verdicts board={board_id} "
            f"pass={n_pass} defect={n_def} escalate={n_esc}"
        )
