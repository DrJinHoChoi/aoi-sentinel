"""Append-only label store.

Edge writes labels here as the operator interacts. Trainer pulls them on
its own cadence (typically nightly). Sync is via SQLite-over-shared-folder
or HTTP — both supported. SQLite chosen as the on-disk format because it
gives us atomic appends and trivial `SELECT ... WHERE id > last_seen`.
"""
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    board_id TEXT NOT NULL,
    ref_des TEXT NOT NULL,
    vendor TEXT NOT NULL,
    line_id TEXT,
    timestamp TEXT NOT NULL,
    image_path TEXT NOT NULL,
    height_map_path TEXT,
    vendor_call TEXT NOT NULL,
    vendor_defect_type TEXT,
    engine_action TEXT NOT NULL,
    engine_confidence REAL NOT NULL,
    operator_label TEXT NOT NULL,    -- 'TRUE_DEFECT' | 'FALSE_CALL' | 'UNSURE'
    operator_id TEXT,
    model_version TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_labels_id ON labels(id);
CREATE INDEX IF NOT EXISTS idx_labels_ts ON labels(timestamp);
"""


@dataclass
class LabelRecord:
    board_id: str
    ref_des: str
    vendor: str
    line_id: str | None
    timestamp: datetime
    image_path: str
    height_map_path: str | None
    vendor_call: str
    vendor_defect_type: str | None
    engine_action: str
    engine_confidence: float
    operator_label: str
    operator_id: str | None
    model_version: str


class LabelQueue:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ write

    def append(self, rec: LabelRecord) -> int:
        with self._conn() as c:
            cur = c.execute(
                """INSERT INTO labels (
                    board_id, ref_des, vendor, line_id, timestamp,
                    image_path, height_map_path,
                    vendor_call, vendor_defect_type,
                    engine_action, engine_confidence,
                    operator_label, operator_id, model_version
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    rec.board_id, rec.ref_des, rec.vendor, rec.line_id,
                    rec.timestamp.isoformat(),
                    rec.image_path, rec.height_map_path,
                    rec.vendor_call, rec.vendor_defect_type,
                    rec.engine_action, rec.engine_confidence,
                    rec.operator_label, rec.operator_id, rec.model_version,
                ),
            )
            return cur.lastrowid

    # ------------------------------------------------------------------ read

    def stream_since(self, last_id: int = 0) -> Iterator[LabelRecord]:
        with self._conn() as c:
            for row in c.execute(
                "SELECT * FROM labels WHERE id > ? ORDER BY id ASC", (last_id,)
            ):
                yield LabelRecord(
                    board_id=row["board_id"],
                    ref_des=row["ref_des"],
                    vendor=row["vendor"],
                    line_id=row["line_id"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    image_path=row["image_path"],
                    height_map_path=row["height_map_path"],
                    vendor_call=row["vendor_call"],
                    vendor_defect_type=row["vendor_defect_type"],
                    engine_action=row["engine_action"],
                    engine_confidence=row["engine_confidence"],
                    operator_label=row["operator_label"],
                    operator_id=row["operator_id"],
                    model_version=row["model_version"],
                )

    def count(self) -> int:
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM labels").fetchone()[0]

    def latest_id(self) -> int:
        with self._conn() as c:
            row = c.execute("SELECT COALESCE(MAX(id), 0) FROM labels").fetchone()
            return int(row[0])
