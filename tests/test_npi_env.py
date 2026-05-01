import numpy as np

from aoi_sentinel.sim import NpiEnv
from aoi_sentinel.sim.cost import (
    ACTION_DEFECT,
    ACTION_ESCALATE,
    ACTION_PASS,
)


def _toy_env(n: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    images = (rng.integers(0, 256, size=(n, 32, 32, 3), dtype=np.uint8))
    labels = rng.integers(0, 2, size=(n,), dtype=np.int64)
    return NpiEnv(images=images, labels=labels, history_length=8, episode_length=n, seed=seed)


def test_reset_and_step_shapes():
    env = _toy_env()
    obs, _ = env.reset(seed=0)
    assert obs["image"].shape == (32, 32, 3)
    assert obs["history"].shape == (8, 5)
    obs, r, term, trunc, info = env.step(ACTION_ESCALATE)
    assert "is_escape" in info
    assert info["is_escape"] in (0, 1)
    assert isinstance(r, float)
    assert not term  # n=32, episode_length=32, but step idx is now 1


def test_escape_only_when_pass_on_defect():
    env = _toy_env()
    env.reset(seed=0)
    # ESCALATE never escapes
    _, _, _, _, info = env.step(ACTION_ESCALATE)
    assert info["is_escape"] == 0
    # DEFECT never escapes
    _, _, _, _, info = env.step(ACTION_DEFECT)
    assert info["is_escape"] == 0


def test_episode_terminates():
    env = _toy_env(n=4)
    env.reset(seed=1)
    done = False
    for _ in range(4):
        _, _, term, trunc, _ = env.step(ACTION_ESCALATE)
        done = term or trunc
    assert done


def test_history_grows():
    env = _toy_env(n=10)
    obs, _ = env.reset(seed=2)
    obs2, _, _, _, _ = env.step(ACTION_PASS)
    # Right-aligned: last row should now be filled (mask>=0)
    assert obs2["history"][-1, 2] == ACTION_PASS  # action recorded
