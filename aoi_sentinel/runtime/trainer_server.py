"""Trainer server — runs on the on-prem trainer box (DGX Spark).

Loop:
    every N hours / every M new labels:
        pull new labels from each edge's LabelQueue
        train a candidate (board-wise split, cost-sensitive focal loss)
        evaluate against the incumbent on a held-out replay
        if safety_gate.evaluate(...) passes → registry.promote(new_version)
        else → discard candidate, log reasons, keep the incumbent
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from aoi_sentinel.runtime.label_queue import LabelQueue, LabelRecord
from aoi_sentinel.runtime.model_registry import ModelHandle, ModelRegistry
from aoi_sentinel.runtime.safety_gate import (
    CandidateScore,
    GateConfig,
    GateResult,
    evaluate as gate_evaluate,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TrainerConfig:
    label_dbs: list[str]                  # one per edge
    model_root: str                       # ModelRegistry path (shared with edges)
    work_dir: str                         # tmp space for candidate training runs
    image_root: str = "."                 # base path for images referenced in label rows

    train_every_seconds: int = 6 * 3600   # default: 4× per day
    min_new_labels: int = 1_000           # don't retrain below this
    holdout_size: int = 5_000             # last-N labels reserved for holdout
    encoder_size: str = "small"           # MobileNetV3 family

    # Training inner loop
    epochs: int = 3
    batch_size: int = 32
    lr: float = 3.0e-4

    # Inferencer threshold tuned downstream
    threshold: float = 0.30
    abstain_band: float = 0.10

    gate: GateConfig | None = None


# ---------------------------------------------------------------------------
# Public entry — long-running loop
# ---------------------------------------------------------------------------


def run_trainer(cfg: TrainerConfig) -> None:
    registry = ModelRegistry(cfg.model_root)
    queues = [LabelQueue(p) for p in cfg.label_dbs]
    last_seen: dict[Path, int] = {q.db_path: 0 for q in queues}

    while True:
        ok = train_one_round(cfg, registry, queues, last_seen)
        if not ok:
            time.sleep(min(cfg.train_every_seconds, 600))
            continue
        time.sleep(cfg.train_every_seconds)


def train_one_round(
    cfg: TrainerConfig,
    registry: ModelRegistry,
    queues: list[LabelQueue],
    last_seen: dict[Path, int],
) -> bool:
    """Run a single train→evaluate→(maybe) promote cycle. Returns True if a
    candidate was actually trained (regardless of promotion outcome)."""
    new_total = sum(q.latest_id() - last_seen[q.db_path] for q in queues)
    print(f"[trainer] new labels available: {new_total}")
    if new_total < cfg.min_new_labels:
        return False

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    candidate_version = f"cand-{ts}"
    print(f"[trainer] training candidate {candidate_version}")

    train_set, holdout = _prepare_data(queues, last_seen, cfg.holdout_size)
    if not train_set:
        print("[trainer] no usable training rows after filtering — abort round")
        return False

    weights, conf = _train_candidate(cfg, candidate_version, train_set)
    cand_score = _evaluate(weights, conf, holdout, cfg)
    inc_score = _evaluate_incumbent(registry, holdout, cfg)

    result: GateResult = gate_evaluate(cand_score, inc_score, cfg.gate)

    if result.passed:
        handle = registry.stage(
            candidate_version,
            weights,
            conf,
            metadata={
                "train_size": cand_score.n,
                "candidate_score": cand_score.__dict__,
                "incumbent_score": inc_score.__dict__,
            },
        )
        registry.promote(handle.version)
        print(f"[trainer] PROMOTED {handle.version}")
    else:
        print(f"[trainer] gate rejected: {result.reasons}")

    for q in queues:
        last_seen[q.db_path] = q.latest_id()
    return True


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def _prepare_data(
    queues: list[LabelQueue],
    last_seen: dict[Path, int],
    holdout_size: int,
) -> tuple[list[LabelRecord], list[LabelRecord]]:
    """Pull all available labels, take the most recent `holdout_size` rows
    by timestamp as the holdout, and use the rest as training data.

    The candidate must NOT be trained on the holdout — that is why we use a
    *temporal* split here rather than a random one. A model that overfits
    to "recent" data is exactly what we want to catch.
    """
    from aoi_sentinel.models.classifier.dataset import filter_labelable

    rows: list[LabelRecord] = []
    for q in queues:
        rows.extend(q.stream_since(0))
    if not rows:
        return [], []

    rows = filter_labelable(rows)
    rows.sort(key=lambda r: r.timestamp)

    if len(rows) <= holdout_size:
        holdout = rows
        train = []
    else:
        cut = len(rows) - holdout_size
        train = rows[:cut]
        holdout = rows[cut:]
    return train, holdout


# ---------------------------------------------------------------------------
# Candidate training
# ---------------------------------------------------------------------------


def _train_candidate(
    cfg: TrainerConfig,
    version: str,
    records: list[LabelRecord],
) -> tuple[Path, Path]:
    """Train a fresh classifier on `records`; return (weights_path, config_path)."""
    from aoi_sentinel.models.classifier.dataset import board_wise_split
    from aoi_sentinel.models.classifier.train import TrainConfig, train_classifier

    out = Path(cfg.work_dir) / version
    out.mkdir(parents=True, exist_ok=True)

    split = board_wise_split(records, holdout_fraction=0.15)
    if not split.train:
        # All boards landed in holdout — fall back to no-val-set training.
        split.train = records
        split.holdout = []

    tc = TrainConfig(
        encoder_size=cfg.encoder_size,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        image_root=cfg.image_root,
    )
    return train_classifier(split.train, split.holdout, tc, out)


# ---------------------------------------------------------------------------
# Holdout evaluation
# ---------------------------------------------------------------------------


def _evaluate(
    weights_path: Path,
    config_path: Path,
    records: list[LabelRecord],
    cfg: TrainerConfig,
) -> CandidateScore:
    """Run inference over the holdout, score with the safety-gate metrics."""
    from aoi_sentinel.models.classifier.infer import Inferencer
    return _score_with_inferencer(
        Inferencer(
            weights_path,
            config_path,
            threshold=cfg.threshold,
            abstain_band=cfg.abstain_band,
        ),
        records,
        image_root=Path(cfg.image_root),
    )


def _evaluate_incumbent(
    registry: ModelRegistry,
    records: list[LabelRecord],
    cfg: TrainerConfig,
) -> CandidateScore:
    """Score the currently-promoted model. Returns a strict-ceiling score
    when there is no incumbent (first-ever round) so any candidate passes."""
    handle = registry.current()
    if handle is None:
        return CandidateScore(
            n=len(records),
            escapes=0,
            fc_rate=1.0,            # baseline assumption: every Saki call is a false call
            escalate_rate=1.0,      # baseline policy: ESCALATE everything
        )

    from aoi_sentinel.models.classifier.infer import Inferencer
    return _score_with_inferencer(
        Inferencer(
            handle.weights_path,
            handle.config_path,
            threshold=cfg.threshold,
            abstain_band=cfg.abstain_band,
        ),
        records,
        image_root=Path(cfg.image_root),
    )


def _score_with_inferencer(inferencer, records: list[LabelRecord], image_root: Path) -> CandidateScore:
    if not records:
        return CandidateScore(n=0, escapes=0, fc_rate=0.0, escalate_rate=0.0)

    import cv2
    n = 0; escapes = 0; fc_calls = 0; escalations = 0
    for r in records:
        if r.operator_label not in {"TRUE_DEFECT", "FALSE_CALL"}:
            continue
        path = Path(r.image_path)
        if not path.is_absolute():
            path = image_root / path
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        action, _ = inferencer(img)
        n += 1
        if action == "ESCALATE":
            escalations += 1
        elif action == "PASS" and r.operator_label == "TRUE_DEFECT":
            escapes += 1
        elif action == "DEFECT" and r.operator_label == "FALSE_CALL":
            fc_calls += 1
    if n == 0:
        return CandidateScore(n=0, escapes=0, fc_rate=0.0, escalate_rate=0.0)
    return CandidateScore(
        n=n,
        escapes=escapes,
        fc_rate=fc_calls / n,
        escalate_rate=escalations / n,
    )
