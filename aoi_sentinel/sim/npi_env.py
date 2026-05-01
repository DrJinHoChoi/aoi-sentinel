"""NPI streaming env.

A Saki-line surrogate built on top of a labeled benchmark stream. Each step:

    1. Env yields the next ROI image + recent inspection-history slice
    2. Agent picks an action ∈ {DEFECT, PASS, ESCALATE}
    3. The oracle reveals the ground-truth label
       (in real life this only happens on ESCALATE; in sim we always get it for cost computation,
        but we mask it out of the agent's history if action ∉ {ESCALATE})
    4. Cost is computed; safety-cost = 1 iff escape

The history buffer encodes the temporal context the Mamba sequence encoder consumes.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from aoi_sentinel.sim.cost import (
    ACTION_DEFECT,
    ACTION_ESCALATE,
    ACTION_PASS,
    LABEL_TRUE_DEFECT,
    CostMatrix,
    default_cost_matrix,
)
from aoi_sentinel.sim.label_oracle import GroundTruthOracle, LabelOracle


@dataclass
class HistoryEntry:
    """One step of inspection history exposed to the sequence encoder."""

    image_idx: int        # index into the frozen image-feature cache
    saki_call: int        # 1 if Saki flagged
    action_taken: int     # 0/1/2
    label_revealed: int   # -1 if not revealed (action != ESCALATE), else 0/1


class NpiEnv(gym.Env):
    """Streaming inspection env for online cost-sensitive RL.

    Observations are deliberately split:
      - obs["image"]    — current ROI image (uint8 HWC)
      - obs["history"]  — fixed-length history tensor (L, 5):
            [image_idx_relative, saki_call, action, label_revealed_mask, label_value]
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        images: np.ndarray,                # (N, H, W, C) uint8
        labels: np.ndarray,                # (N,) int — 0=false_call, 1=true_defect
        saki_calls: np.ndarray | None = None,  # defaults to all-ones (Saki flagged everything)
        history_length: int = 512,
        cost: CostMatrix | None = None,
        oracle: LabelOracle | None = None,
        episode_length: int | None = None,  # default: walk through full stream once
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if images.ndim != 4:
            raise ValueError("images must be (N,H,W,C)")
        if len(images) != len(labels):
            raise ValueError("images/labels length mismatch")

        self.images = images
        self.labels = labels.astype(np.int64)
        self.saki_calls = saki_calls if saki_calls is not None else np.ones(len(labels), dtype=np.int64)
        self.cost = cost or default_cost_matrix()
        self.oracle = oracle or GroundTruthOracle(self.labels)
        self.history_length = history_length
        self.episode_length = episode_length or len(images)
        self._rng = np.random.default_rng(seed)

        h, w, c = images.shape[1:]
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(0, 255, shape=(h, w, c), dtype=np.uint8),
                "history": spaces.Box(
                    low=-1.0,
                    high=float(len(images)),
                    shape=(history_length, 5),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = spaces.Discrete(3)

        self._order: np.ndarray = np.empty(0, dtype=np.int64)
        self._step_idx: int = 0
        self._history: deque[HistoryEntry] = deque(maxlen=history_length)
        self._cumulative_cost: float = 0.0
        self._cumulative_escape: int = 0

    # ------------------------------------------------------------------ gym API

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # NPI = arrivals are time-ordered; for sim we keep insertion order
        # but allow shuffling to vary lots between epochs.
        n = min(self.episode_length, len(self.images))
        self._order = self._rng.permutation(len(self.images))[:n]
        self._step_idx = 0
        self._history.clear()
        self._cumulative_cost = 0.0
        self._cumulative_escape = 0
        return self._observe(), {}

    def step(self, action: int):
        sample_id = int(self._order[self._step_idx])
        label = self.oracle.reveal(sample_id)

        cost = float(self.cost.matrix()[label, action])
        is_escape = self.cost.is_escape(label, action)

        # Reward = -cost. Safety cost is the escape indicator (consumed by Lagrangian PPO).
        reward = -cost
        info = {
            "label": label,
            "action": action,
            "cost": cost,
            "is_escape": int(is_escape),
            "sample_id": sample_id,
            "cumulative_cost": self._cumulative_cost + cost,
            "cumulative_escape": self._cumulative_escape + int(is_escape),
            "step": self._step_idx,
        }

        self._cumulative_cost += cost
        self._cumulative_escape += int(is_escape)

        # Append to history. In the real line we would only see the label when
        # we ESCALATE; we mirror that to keep the agent honest.
        revealed = label if action == ACTION_ESCALATE else -1
        self._history.append(
            HistoryEntry(
                image_idx=sample_id,
                saki_call=int(self.saki_calls[sample_id]),
                action_taken=int(action),
                label_revealed=int(revealed),
            )
        )

        self._step_idx += 1
        terminated = self._step_idx >= len(self._order)
        truncated = False
        return self._observe(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ helpers

    def _observe(self) -> dict[str, np.ndarray]:
        if self._step_idx >= len(self._order):
            # Terminal observation — just return zeros; the trainer ignores it.
            h, w, c = self.images.shape[1:]
            return {
                "image": np.zeros((h, w, c), dtype=np.uint8),
                "history": self._history_tensor(),
            }
        sample_id = int(self._order[self._step_idx])
        return {
            "image": self.images[sample_id],
            "history": self._history_tensor(),
        }

    def _history_tensor(self) -> np.ndarray:
        out = np.full((self.history_length, 5), -1.0, dtype=np.float32)
        # Right-pad: most recent entry at the end.
        start = self.history_length - len(self._history)
        for i, e in enumerate(self._history):
            mask = 1.0 if e.label_revealed >= 0 else 0.0
            out[start + i] = (
                float(e.image_idx),
                float(e.saki_call),
                float(e.action_taken),
                mask,
                float(max(e.label_revealed, 0)),
            )
        return out

    @property
    def cumulative_cost(self) -> float:
        return self._cumulative_cost

    @property
    def cumulative_escape(self) -> int:
        return self._cumulative_escape
