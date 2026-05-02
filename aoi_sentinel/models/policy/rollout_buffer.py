"""On-policy rollout buffer with dual GAE (reward + safety cost).

Karpathy: simplest thing that works. The math is plain GAE-λ applied
twice — once for reward, once for safety cost. The Lagrangian PPO update
combines them through λ (the dual variable) so we don't need a separate
constrained-optimisation library.

Memory: O(rollout_steps × obs_size). The sequence encoder reads
`history` slices of length L, so per-step storage is dominated by the
ROI image (224×224×3 = 150 KB at uint8). 256-step rollouts → ~38 MB,
well within a T4's free RAM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TrajectoryStats:
    n_steps: int = 0
    return_: float = 0.0           # cumulative reward
    cost_return: float = 0.0       # cumulative safety-cost (escape count)
    n_escapes: int = 0
    n_false_calls: int = 0
    n_escalations: int = 0


# ---------------------------------------------------------------------------
# Buffer
# ---------------------------------------------------------------------------


@dataclass
class RolloutBuffer:
    """Fixed-size buffer for one PPO rollout.

    All arrays are pre-allocated np.ndarray; we hand them to torch on
    update. Storing as numpy keeps memory lean and avoids GPU OOM on
    long rollouts (image stack is the largest tensor).
    """

    capacity: int
    image_shape: tuple[int, int, int]      # (H, W, C)
    history_length: int
    history_dim: int = 5
    gamma: float = 0.99
    gae_lambda: float = 0.95

    images:    np.ndarray = field(init=False)
    histories: np.ndarray = field(init=False)
    actions:   np.ndarray = field(init=False)
    log_probs: np.ndarray = field(init=False)
    values:    np.ndarray = field(init=False)
    cost_values: np.ndarray = field(init=False)
    rewards:   np.ndarray = field(init=False)
    costs:     np.ndarray = field(init=False)
    dones:     np.ndarray = field(init=False)

    advantages:        np.ndarray = field(init=False)
    cost_advantages:   np.ndarray = field(init=False)
    returns:           np.ndarray = field(init=False)
    cost_returns:      np.ndarray = field(init=False)

    _ptr: int = 0
    _last_value: float = 0.0
    _last_cost_value: float = 0.0

    def __post_init__(self) -> None:
        h, w, c = self.image_shape
        self.images = np.zeros((self.capacity, h, w, c), dtype=np.uint8)
        self.histories = np.zeros((self.capacity, self.history_length, self.history_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.log_probs = np.zeros(self.capacity, dtype=np.float32)
        self.values = np.zeros(self.capacity, dtype=np.float32)
        self.cost_values = np.zeros(self.capacity, dtype=np.float32)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.costs = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.advantages = np.zeros(self.capacity, dtype=np.float32)
        self.cost_advantages = np.zeros(self.capacity, dtype=np.float32)
        self.returns = np.zeros(self.capacity, dtype=np.float32)
        self.cost_returns = np.zeros(self.capacity, dtype=np.float32)

    # --------------------------------------------------------- mutation API

    def add(
        self,
        image: np.ndarray,
        history: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
        cost_value: float,
        reward: float,
        cost: float,
        done: bool,
    ) -> None:
        if self._ptr >= self.capacity:
            raise RuntimeError("rollout buffer full")
        i = self._ptr
        self.images[i] = image
        self.histories[i] = history
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.values[i] = value
        self.cost_values[i] = cost_value
        self.rewards[i] = reward
        self.costs[i] = cost
        self.dones[i] = float(done)
        self._ptr += 1

    def set_terminal_values(self, last_value: float, last_cost_value: float) -> None:
        """Bootstrap values for the partial step at the end of the rollout."""
        self._last_value = float(last_value)
        self._last_cost_value = float(last_cost_value)

    # ----------------------------------------------------------- GAE compute

    def compute_advantages(self, normalise: bool = True) -> None:
        """Standard GAE-λ on rewards; same recipe on safety costs.

        We deliberately do NOT normalise the cost advantages — the
        Lagrangian update uses the raw cost-return mean as the
        constraint signal, and normalising would erase its scale.
        """
        n = self._ptr
        # Reward advantages
        gae = 0.0
        for t in reversed(range(n)):
            next_value = self._last_value if t == n - 1 else self.values[t + 1]
            next_nonterminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + self.gamma * next_value * next_nonterminal - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * next_nonterminal * gae
            self.advantages[t] = gae
        self.returns[:n] = self.advantages[:n] + self.values[:n]

        # Cost advantages — same recursion, same gamma
        gae_c = 0.0
        for t in reversed(range(n)):
            next_cv = self._last_cost_value if t == n - 1 else self.cost_values[t + 1]
            next_nonterminal = 1.0 - self.dones[t]
            delta_c = self.costs[t] + self.gamma * next_cv * next_nonterminal - self.cost_values[t]
            gae_c = delta_c + self.gamma * self.gae_lambda * next_nonterminal * gae_c
            self.cost_advantages[t] = gae_c
        self.cost_returns[:n] = self.cost_advantages[:n] + self.cost_values[:n]

        if normalise and n > 1:
            adv = self.advantages[:n]
            self.advantages[:n] = (adv - adv.mean()) / (adv.std() + 1e-8)

    # ----------------------------------------------------------- iteration

    def __len__(self) -> int:
        return self._ptr

    def reset(self) -> None:
        self._ptr = 0
        self._last_value = 0.0
        self._last_cost_value = 0.0

    def to_torch(self, device: Any):
        """Materialise the filled portion of the buffer as torch tensors.
        Caller is responsible for image normalisation."""
        import torch

        n = self._ptr
        return {
            "images":          torch.from_numpy(self.images[:n]).to(device),
            "histories":       torch.from_numpy(self.histories[:n]).to(device),
            "actions":         torch.from_numpy(self.actions[:n]).to(device),
            "log_probs":       torch.from_numpy(self.log_probs[:n]).to(device),
            "values":          torch.from_numpy(self.values[:n]).to(device),
            "cost_values":     torch.from_numpy(self.cost_values[:n]).to(device),
            "advantages":      torch.from_numpy(self.advantages[:n]).to(device),
            "cost_advantages": torch.from_numpy(self.cost_advantages[:n]).to(device),
            "returns":         torch.from_numpy(self.returns[:n]).to(device),
            "cost_returns":    torch.from_numpy(self.cost_returns[:n]).to(device),
        }
