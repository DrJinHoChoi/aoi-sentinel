"""Koh Young AOI adapter (Zenith / Aspire / Neptune families).

Koh Young (Korean) dominant in 3D AOI/SPI; KSDB-Studio is the central data store.
Common output paths in customer sites:

  1. **CSV/XML export from KSDB-Studio** — easiest, file-based, no SDK
  2. **K-API REST** (newer KSMART releases) — JSON over HTTPS
  3. **Direct DB read** — Postgres/MSSQL behind KSDB-Studio (vendor doesn't bless this)

This adapter starts with (1) since it works on every Koh Young site without
SDK licensing. (2) is added as a config flag once we get K-API docs from a
customer. (3) we avoid for support reasons.

Until a real Koh Young dump is in hand, the parser raises NotImplementedError.
The CSV/XML schema is documented in `docs/vendor_adapter_guide.md`.
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Iterator, Literal

from aoi_sentinel.adapters.base import (
    CommonInspection,
    ComponentInspection,
    Verdict,
    register,
)


KohYoungBackend = Literal["ksdb_export", "k_api"]


@register("koh_young")
class KohYoungAdapter:
    name: str

    def __init__(
        self,
        backend: KohYoungBackend = "ksdb_export",
        poll_interval_s: float = 2.0,
    ) -> None:
        self.backend = backend
        self.poll_interval_s = poll_interval_s
        self._seen: set[str] = set()

    # ------------------------------------------------------------------ watch

    def watch(self, source: str | Path) -> Iterator[CommonInspection]:
        if self.backend == "ksdb_export":
            yield from self._watch_ksdb_export(Path(source))
        elif self.backend == "k_api":
            yield from self._watch_k_api(str(source))
        else:
            raise ValueError(f"unknown backend: {self.backend}")

    def _watch_ksdb_export(self, root: Path) -> Iterator[CommonInspection]:
        """Watch a folder where KSDB-Studio drops per-board CSV+image bundles."""
        if not root.exists():
            raise FileNotFoundError(root)
        while True:
            for csv_path in sorted(root.rglob("*.csv")):
                if str(csv_path) in self._seen:
                    continue
                try:
                    insp = self._parse_ksdb_csv(csv_path)
                except NotImplementedError:
                    raise
                except Exception as e:  # noqa: BLE001
                    print(f"[koh_young] skip malformed {csv_path}: {e}")
                    self._seen.add(str(csv_path))
                    continue
                self._seen.add(str(csv_path))
                yield insp
            time.sleep(self.poll_interval_s)

    def _watch_k_api(self, base_url: str) -> Iterator[CommonInspection]:
        """Long-poll the Koh Young K-API for new inspection events."""
        raise NotImplementedError(
            "K-API integration deferred until we have docs from a customer site"
        )

    # ------------------------------------------------------------------ parse

    def _parse_ksdb_csv(self, csv_path: Path) -> CommonInspection:
        """Parse one board's KSDB-Studio export.

        TODO: fill once we have a real sample. Expected columns based on
        public KSMART screenshots and customer reports:

            Barcode, InspectionTime, LineName, LotNo,
            RefDesignator, X, Y, Width, Height,
            Image2D, Image3D, Judgement, DefectCode

        Image paths are usually relative to the CSV.
        """
        raise NotImplementedError(
            "Koh Young KSDB CSV parser pending — provide a 1-board sample export "
            "and we wire it up. Schema in docs/vendor_adapter_guide.md."
        )

    # ---------------------------------------------------------------- verdict

    def push_verdict(self, board_id: str, verdicts: list[Verdict]) -> None:
        """KSMART integration via K-API or sidecar JSON. Phased rollout per site."""
        print(f"[koh_young] verdicts board={board_id} ({len(verdicts)}) — sidecar TODO")
