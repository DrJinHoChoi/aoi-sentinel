"""Trainer server — runs on the on-prem trainer box (DGX Spark).

Loop:
    every N hours / every M new labels:
        pull new labels from each edge's LabelQueue
        retrain (continual fine-tune) on accumulated data
        evaluate against incumbent on hold-out
        if safety_gate.evaluate(...) passes → registry.promote(new_version)
        else → discard candidate, log why
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from aoi_sentinel.runtime.label_queue import LabelQueue
from aoi_sentinel.runtime.model_registry import ModelRegistry
from aoi_sentinel.runtime.safety_gate import (
    CandidateScore,
    GateConfig,
    GateResult,
    evaluate,
)


@dataclass
class TrainerConfig:
    label_dbs: list[str]                  # one per edge
    model_root: str                       # ModelRegistry path (shared with edges)
    work_dir: str                         # tmp space for candidate training runs
    train_every_seconds: int = 6 * 3600   # default: 4× per day
    min_new_labels: int = 1_000           # don't retrain below this
    holdout_size: int = 5_000             # evaluate against this many recent labels
    gate: GateConfig | None = None


def run_trainer(cfg: TrainerConfig) -> None:
    registry = ModelRegistry(cfg.model_root)
    queues = [LabelQueue(p) for p in cfg.label_dbs]
    last_seen = {q.db_path: 0 for q in queues}

    while True:
        new_total = sum(q.latest_id() - last_seen[q.db_path] for q in queues)
        print(f"[trainer] new labels available: {new_total}")

        if new_total >= cfg.min_new_labels:
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            candidate_version = f"cand-{ts}"
            print(f"[trainer] training candidate {candidate_version}")

            train_set, holdout = _prepare_data(queues, last_seen, cfg.holdout_size)
            weights, conf = _train_candidate(
                cfg.work_dir,
                candidate_version,
                train_set,
                init_from=registry.current(),
            )

            cand_score = _evaluate(weights, conf, holdout)
            inc_score = _evaluate_incumbent(registry, holdout)
            result: GateResult = evaluate(cand_score, inc_score, cfg.gate)

            if result.passed:
                handle = registry.stage(candidate_version, weights, conf, metadata={
                    "train_size": cand_score.n,
                    "candidate_score": cand_score.__dict__,
                    "incumbent_score": inc_score.__dict__,
                })
                registry.promote(handle.version)
                print(f"[trainer] PROMOTED {handle.version}")
            else:
                print(f"[trainer] gate rejected: {result.reasons}")

            for q in queues:
                last_seen[q.db_path] = q.latest_id()

        time.sleep(cfg.train_every_seconds)


# ---------------------------------------------------------------------------
# internal stubs — to be filled in as the training loop solidifies
# ---------------------------------------------------------------------------


def _prepare_data(queues, last_seen, holdout_size):
    raise NotImplementedError


def _train_candidate(work_dir, version, train_set, init_from):
    raise NotImplementedError


def _evaluate(weights_path, config_path, holdout) -> CandidateScore:
    raise NotImplementedError


def _evaluate_incumbent(registry: ModelRegistry, holdout) -> CandidateScore:
    raise NotImplementedError
