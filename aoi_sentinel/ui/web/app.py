"""FastAPI app for the operator UI."""
from __future__ import annotations

import asyncio
import json
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque

from fastapi import FastAPI, Form, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from aoi_sentinel.runtime.label_queue import LabelQueue, LabelRecord
from aoi_sentinel.runtime.modes import Mode

# ---------------------------------------------------------------------------
# App + paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=BASE_DIR / "templates")

app = FastAPI(title="aoi-sentinel UI", version="0.1.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# In-memory state — single edge box, single line. Multi-line later.
_inbox: Deque[dict] = deque(maxlen=200)
_websockets: list[WebSocket] = []
_mode: Mode = Mode.SHADOW
_label_queue: LabelQueue | None = None
_kpi: dict = {"boards": 0, "false_calls": 0, "escapes": 0, "escalations": 0}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "mode": _mode.value, "kpi": _kpi},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "mode": _mode.value, "queue_depth": len(_inbox)}


@app.post("/decide", response_class=HTMLResponse)
async def decide(
    request: Request,
    board_id: str = Form(...),
    ref_des: str = Form(...),
    operator_label: str = Form(...),  # 'TRUE_DEFECT' | 'FALSE_CALL' | 'UNSURE'
    operator_id: str = Form(default=""),
):
    """Operator submits a decision for one ROI."""
    if _label_queue is None:
        return HTMLResponse("label queue not configured", status_code=503)

    # Pull the originating event off the inbox (best-effort match by board+ref_des).
    event = next(
        (e for e in _inbox if e["board_id"] == board_id and e["ref_des"] == ref_des),
        None,
    )
    if event is None:
        return HTMLResponse("event not found", status_code=404)

    rec = LabelRecord(
        board_id=board_id,
        ref_des=ref_des,
        vendor=event["vendor"],
        line_id=event.get("line_id"),
        timestamp=datetime.utcnow(),
        image_path=event["image_path"],
        height_map_path=event.get("height_map_path"),
        vendor_call=event["vendor_call"],
        vendor_defect_type=event.get("vendor_defect_type"),
        engine_action=event["engine_action"],
        engine_confidence=float(event["engine_confidence"]),
        operator_label=operator_label,
        operator_id=operator_id or None,
        model_version=event["model_version"],
    )
    _label_queue.append(rec)

    # KPI counters
    _kpi["boards"] += 1
    if event["engine_action"] == "PASS" and operator_label == "TRUE_DEFECT":
        _kpi["escapes"] += 1
    if event["engine_action"] == "DEFECT" and operator_label == "FALSE_CALL":
        _kpi["false_calls"] += 1
    if event["engine_action"] == "ESCALATE":
        _kpi["escalations"] += 1

    return HTMLResponse(f"<div class='ack'>recorded — {ref_des} → {operator_label}</div>")


@app.websocket("/ws/inbox")
async def ws_inbox(ws: WebSocket):
    """Push new ROIs to the operator as they arrive from the edge."""
    await ws.accept()
    _websockets.append(ws)
    try:
        # Send any buffered events on connect
        for evt in list(_inbox):
            await ws.send_text(json.dumps(evt))
        while True:
            await asyncio.sleep(60)  # heartbeat — clients re-render via HTMX swap
    except WebSocketDisconnect:
        pass
    finally:
        if ws in _websockets:
            _websockets.remove(ws)


# ---------------------------------------------------------------------------
# Edge → UI ingestion (called by edge.py over HTTP or in-process queue)
# ---------------------------------------------------------------------------


async def push_roi_event(event: dict) -> None:
    """Edge daemon calls this when a new ROI needs operator attention."""
    _inbox.append(event)
    payload = json.dumps(event)
    dead: list[WebSocket] = []
    for ws in _websockets:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in _websockets:
            _websockets.remove(ws)


def configure(*, label_queue: LabelQueue, mode: Mode = Mode.SHADOW) -> None:
    """One-shot config from the embedding process."""
    global _label_queue, _mode
    _label_queue = label_queue
    _mode = mode
