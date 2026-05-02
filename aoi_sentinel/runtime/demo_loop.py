"""Demo loop — POC runtime that ties adapter + UI on the Jetson Nano box.

Run this directly:
    python -m aoi_sentinel.runtime.demo_loop

Environment variables (set by the systemd unit):
    AOI_DEMO_WATCH        folder watched for new CSV bundles  (default /tmp/aoi-watch)
    AOI_DEMO_PORT         UI port                              (default 8080)
    AOI_DEMO_LIGHTWEIGHT  if set, skip torch/timm import       (POC default)
    AOI_DEMO_LABEL_DB     SQLite path                          (default ~/aoi_demo_labels.db)

What this does:
    1. Boots the FastAPI operator UI on AOI_DEMO_PORT.
    2. Spins up a generic_csv adapter watching AOI_DEMO_WATCH.
    3. For each component the adapter yields, push a card event to the UI.
    4. When the operator clicks PASS / DEFECT / 모름, the UI writes a label
       row into the local SQLite queue.
    5. KPI tiles in the UI update from the queue's rolling stats.
"""
from __future__ import annotations

import asyncio
import base64
import os
import threading
from datetime import datetime
from pathlib import Path

import uvicorn

from aoi_sentinel.adapters import make_adapter
from aoi_sentinel.runtime.label_queue import LabelQueue
from aoi_sentinel.runtime.modes import Mode
from aoi_sentinel.ui.web.app import app, configure, push_roi_event


def _read_image_b64(path: Path) -> str:
    """Inline the image into the websocket payload so the browser can render
    without a separate /static endpoint. Fine for POC scale (≤ 100 cards/board)."""
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


async def _adapter_loop(adapter, watch: Path, model_version: str) -> None:
    """Bridge the adapter stream into the UI's websocket inbox."""
    loop = asyncio.get_running_loop()

    def _watch_blocking():
        for inspection in adapter.watch(watch):
            for c in inspection.components:
                # Resolve image path relative to the bundle root
                img_path = watch / c.image_2d if not Path(c.image_2d).is_absolute() else Path(c.image_2d)
                if not img_path.exists():
                    # generic_csv stores paths relative to the CSV file's directory
                    img_path = (Path(inspection.raw_payload_path).parent / c.image_2d) if inspection.raw_payload_path else img_path
                if not img_path.exists():
                    continue

                # POC: no real model. Random-ish but biased on defect_type for demo theatrics.
                conf = 0.55 if (c.vendor_defect_type or "").startswith(("M", "T")) else 0.78
                action = "ESCALATE"  # SHADOW mode — always ask the operator

                event = {
                    "board_id": inspection.board_id,
                    "ref_des": c.ref_des,
                    "vendor": inspection.vendor,
                    "line_id": inspection.line_id,
                    "vendor_call": c.vendor_call,
                    "vendor_defect_type": c.vendor_defect_type,
                    "image_url": _read_image_b64(img_path),
                    "image_path": str(img_path),
                    "engine_action": action,
                    "engine_confidence": conf,
                    "model_version": model_version,
                }
                # Hop back to the asyncio loop for the websocket fan-out.
                asyncio.run_coroutine_threadsafe(push_roi_event(event), loop)

    # Run the (blocking) adapter watcher on a background thread.
    await loop.run_in_executor(None, _watch_blocking)


def main() -> None:
    watch = Path(os.environ.get("AOI_DEMO_WATCH", "/tmp/aoi-watch"))
    port = int(os.environ.get("AOI_DEMO_PORT", "8080"))
    db_path = Path(os.environ.get("AOI_DEMO_LABEL_DB", str(Path.home() / "aoi_demo_labels.db")))
    model_version = os.environ.get("AOI_DEMO_MODEL_VERSION", "poc-stub-v0")

    watch.mkdir(parents=True, exist_ok=True)
    queue = LabelQueue(db_path)
    configure(label_queue=queue, mode=Mode.SHADOW)

    adapter = make_adapter("generic_csv")

    # Patch in the adapter loop as a startup task.
    @app.on_event("startup")
    async def _startup() -> None:
        asyncio.create_task(_adapter_loop(adapter, watch, model_version))

    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    main()
