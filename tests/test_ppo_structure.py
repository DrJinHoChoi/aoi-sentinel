"""Lagrangian PPO structure tests — pure numpy / pure Python.

The actual training loop is exercised by the Colab notebook + bench
script. Here we lock down the math: GAE recursion, λ update direction,
buffer round-trip behaviour. If any of these regress, training will
silently produce nonsense.
"""
from __future__ import annotations

import numpy as np
import pytest

from aoi_sentinel.models.policy.lagrangian_ppo import (
    PPOConfig,
    softplus_lambda,
    update_lambda,
)
from aoi_sentinel.models.policy.rollout_buffer import RolloutBuffer


# ---------------------------------------------------------------- λ helpers

def test_softplus_lambda_nonnegative():
    for raw in (-1000.0, -1.0, 0.0, 1.0, 5.0, 50.0):
        assert softplus_lambda(raw) >= 0.0


def test_update_lambda_grows_when_violating():
    raw = 0.0
    new = update_lambda(raw, mean_cost=0.05, cost_limit=0.001, lr=0.1)
    assert new > raw  # violating budget → λ increases


def test_update_lambda_shrinks_when_satisfying():
    raw = 5.0
    new = update_lambda(raw, mean_cost=0.0, cost_limit=0.01, lr=0.1)
    assert new < raw  # under-budget → λ decreases


def test_update_lambda_capped():
    raw = 100.0
    new = update_lambda(raw, mean_cost=10.0, cost_limit=0.0, lr=10.0, lambda_max=10.0)
    # Visible λ must stay ≤ lambda_max even though the violation is huge.
    assert softplus_lambda(new) <= 10.0 + 1e-3


# ---------------------------------------------------------------- buffer

def _make_buf(n: int = 8, L: int = 4) -> RolloutBuffer:
    buf = RolloutBuffer(
        capacity=n, image_shape=(8, 8, 3), history_length=L,
        gamma=0.9, gae_lambda=0.95,
    )
    rng = np.random.default_rng(0)
    for _ in range(n):
        buf.add(
            image=rng.integers(0, 255, size=(8, 8, 3), dtype=np.uint8),
            history=np.zeros((L, 5), dtype=np.float32),
            action=int(rng.integers(0, 3)),
            log_prob=float(rng.normal()),
            value=float(rng.normal()),
            cost_value=float(rng.normal()),
            reward=float(rng.normal()),
            cost=float(rng.uniform(0, 1)),
            done=False,
        )
    buf.set_terminal_values(0.0, 0.0)
    return buf


def test_buffer_records_capacity_correctly():
    buf = _make_buf(n=5)
    assert len(buf) == 5
    with pytest.raises(RuntimeError):
        # buffer is full — adding more must error
        for _ in range(10):
            buf.add(
                image=np.zeros((8, 8, 3), dtype=np.uint8),
                history=np.zeros((4, 5), dtype=np.float32),
                action=0, log_prob=0.0, value=0.0, cost_value=0.0,
                reward=0.0, cost=0.0, done=False,
            )


def test_gae_returns_match_advantages_plus_values():
    buf = _make_buf(n=6)
    buf.compute_advantages(normalise=False)
    n = len(buf)
    np.testing.assert_allclose(
        buf.returns[:n],
        buf.advantages[:n] + buf.values[:n],
        rtol=1e-6, atol=1e-6,
    )
    np.testing.assert_allclose(
        buf.cost_returns[:n],
        buf.cost_advantages[:n] + buf.cost_values[:n],
        rtol=1e-6, atol=1e-6,
    )


def test_gae_zero_rewards_zero_advantage():
    """If all rewards and values are zero, advantages are zero."""
    buf = RolloutBuffer(capacity=4, image_shape=(8, 8, 3), history_length=2,
                        gamma=0.9, gae_lambda=0.95)
    for _ in range(4):
        buf.add(
            image=np.zeros((8, 8, 3), dtype=np.uint8),
            history=np.zeros((2, 5), dtype=np.float32),
            action=0, log_prob=0.0, value=0.0, cost_value=0.0,
            reward=0.0, cost=0.0, done=False,
        )
    buf.set_terminal_values(0.0, 0.0)
    buf.compute_advantages(normalise=False)
    np.testing.assert_array_equal(buf.advantages[:4], np.zeros(4))
    np.testing.assert_array_equal(buf.cost_advantages[:4], np.zeros(4))


def test_gae_normalisation_zeroes_mean():
    buf = _make_buf(n=8)
    buf.compute_advantages(normalise=True)
    n = len(buf)
    assert abs(float(buf.advantages[:n].mean())) < 1e-6
    assert abs(float(buf.advantages[:n].std()) - 1.0) < 1e-2


def test_buffer_reset_clears_pointer():
    buf = _make_buf(n=5)
    assert len(buf) == 5
    buf.reset()
    assert len(buf) == 0


# ---------------------------------------------------------------- config

def test_ppoconfig_defaults_sane():
    cfg = PPOConfig()
    assert 0 < cfg.cost_limit < 1
    assert 0 < cfg.clip_eps < 1
    assert cfg.gamma <= 1.0 and cfg.gae_lambda <= 1.0
    assert cfg.lambda_lr > 0
