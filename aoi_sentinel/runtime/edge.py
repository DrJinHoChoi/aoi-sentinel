"""Edge daemon — runs on the line box (Jetson Orin Nano).

Loop:
    adapter.watch(folder)
        → for each CommonInspection, for each component:
              run inference
              decide action per current Mode
              push to operator UI queue (via FastAPI websocket)
              record (image_path, engine_action) — operator label arrives later
        → at board completion, send verdicts back via adapter.push_verdict
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from aoi_sentinel.adapters import CommonInspection, Verdict, make_adapter
from aoi_sentinel.runtime.label_queue import LabelQueue
from aoi_sentinel.runtime.model_registry import ModelRegistry
from aoi_sentinel.runtime.modes import Mode


@dataclass
class EdgeConfig:
    vendor: str
    source: str                      # adapter source — folder, URL, OPC-UA
    model_root: str                  # ModelRegistry path
    label_db: str                    # LabelQueue path
    image_cache_dir: str             # where edge stores ROIs for later operator review
    initial_mode: str = "SHADOW"
    image_quality: int = 90          # JPEG quality for cached ROIs


def run_edge(cfg: EdgeConfig) -> None:
    adapter = make_adapter(cfg.vendor)
    registry = ModelRegistry(cfg.model_root)
    queue = LabelQueue(cfg.label_db)
    mode = Mode(cfg.initial_mode)

    handle = registry.current()
    if handle is None:
        raise RuntimeError(
            f"no model in registry at {cfg.model_root}. "
            "Run trainer_server.py first or seed with a Phase-0 checkpoint."
        )

    print(f"[edge] adapter={cfg.vendor} model={handle.version} mode={mode.value}")

    # Lazy import — torch only required if we actually run inference here.
    # On Jetson Nano we may run a TensorRT engine instead; that path is added later.
    inferencer = _build_inferencer(handle.weights_path, handle.config_path)

    cache = Path(cfg.image_cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    for board in adapter.watch(cfg.source):
        verdicts = _process_board(board, inferencer, mode, queue, cache, handle.version)
        adapter.push_verdict(board.board_id, verdicts)


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _process_board(
    board: CommonInspection,
    inferencer,
    mode: Mode,
    queue: LabelQueue,
    cache: Path,
    model_version: str,
) -> list[Verdict]:
    verdicts: list[Verdict] = []
    for c in board.components:
        action, confidence = inferencer(c.image_2d, c.height_map)
        # In SHADOW mode the engine's call is informational only — a thin policy
        # wrapper. The runtime layer above handles the actual operator UI push.
        if mode is Mode.SHADOW:
            shown = "ESCALATE"  # always show to operator for label collection
        elif mode is Mode.ASSIST:
            shown = action if action != "PASS" else "ESCALATE"
        else:  # AUTONOMOUS
            shown = action
        verdicts.append(
            Verdict(
                ref_des=c.ref_des,
                action=shown,
                confidence=confidence,
                rationale=None,
                model_version=model_version,
            )
        )
        # Caching + queue write is the responsibility of the UI layer once it
        # has the operator's response. Edge daemon just emits the candidate.

    return verdicts


def _build_inferencer(weights_path: str, config_path: str):
    """Return a callable (image, height_map) -> (action, confidence).

    On Jetson Nano this may load a TensorRT engine instead. The trainer
    publishes both .pt and .engine artifacts; edge picks whichever is faster.
    """
    # Lazy stub. Filled in once the engine is up to a working v0.
    def _stub(image, height_map):
        time.sleep(0)
        return "ESCALATE", 0.5
    return _stub
