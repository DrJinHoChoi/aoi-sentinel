"""Stage 1 — Lagrangian PPO with Mamba RL on the NPI simulator.

Run via:
    aoi train npi-rl --config configs/stage1_npi_rl_light.yaml

Default backbone is **Mamba** (linear in sequence length L). Set
`backbone: transformer` in the config to swap for the O(L²) baseline —
useful for the bench script and as a Karpathy-grade "must beat baseline"
check before any Mamba claim.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def run(config_path: str | Path) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    out_dir = Path(cfg.get("out_dir", "checkpoints/stage1"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    model, env = _build(cfg)
    trainer = _build_trainer(model, env, cfg)

    backbone = cfg.get("backbone", "mamba")
    print(f"[stage1] backbone={backbone}  rollout={trainer.cfg.rollout_steps}  iters={cfg['iterations']}")
    print(f"[stage1] cost_limit ε={trainer.cfg.cost_limit}  init λ={trainer.lagrange_multiplier:.3f}")

    history = []
    log_every = cfg.get("log_every", 1)
    save_every = cfg.get("save_every", 25)

    for it in range(cfg["iterations"]):
        log = trainer.step()
        history.append({"iter": it, **log})
        if it % log_every == 0:
            print(
                f"iter {it:4d}  λ={log['lambda']:.4f}  "
                f"avg_cost={log.get('mean_cost_return', 0):.5f}  "
                f"reward={log.get('mean_reward', 0):+.3f}  "
                f"escapes={log['ep_escapes']:3d}  fc={log['ep_false_calls']:3d}  esc={log['ep_escalations']:3d}  "
                f"ploss={log['policy_loss']:+.4f}  vloss={log['value_loss']:.4f}  "
                f"cum_cost={env.cumulative_cost:.1f}  cum_escape={env.cumulative_escape}"
            )
        if (it + 1) % save_every == 0:
            _save_ckpt(trainer.model, out_dir / f"stage1_iter{it+1:05d}.pt", cfg, history)

    _save_ckpt(trainer.model, out_dir / "stage1_final.pt", cfg, history)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[stage1] done — {out_dir}")


# ---------------------------------------------------------------------------
# Build env + model
# ---------------------------------------------------------------------------


def _build(cfg: dict):
    """All torch / gymnasium imports live here so importing the module is free."""
    import torch

    from aoi_sentinel.sim import NpiEnv
    from aoi_sentinel.sim.cost import CostMatrix

    cost_cfg = cfg.get("cost", {})
    cost = CostMatrix(
        c_escape=cost_cfg.get("c_escape", 1000.0),
        c_false_call=cost_cfg.get("c_false_call", 1.0),
        c_operator=cost_cfg.get("c_operator", 5.0),
    )

    images, labels = _load_dataset(cfg)
    env_cfg = cfg.get("env", {})
    env = NpiEnv(
        images=images,
        labels=labels,
        history_length=env_cfg.get("history_length", 256),
        cost=cost,
        episode_length=env_cfg.get("episode_length", 1024),
        seed=cfg.get("seed", 42),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(cfg).to(device)
    return model, env


def _build_model(cfg: dict):
    """Mamba (default) or transformer (baseline) actor-critic."""
    backbone = cfg.get("backbone", "mamba")
    image_cfg = cfg["model"]["image"]
    seq_cfg = cfg["model"]["sequence"]
    trunk_hidden = cfg["model"].get("trunk_hidden", 384)

    # Vision encoder is shared. Use the lightweight one — the "Mamba RL is
    # much lighter than transformer" message means we don't waste a
    # 100M-param vision backbone on the policy head.
    from aoi_sentinel.models.lightweight import build_lightweight_encoder
    image_enc = build_lightweight_encoder({
        "size": image_cfg.get("size", "small"),
        "pretrained": image_cfg.get("pretrained", True),
    })
    if image_cfg.get("freeze_backbone", False):
        for p in image_enc.parameters():
            p.requires_grad = False

    if backbone == "transformer":
        from aoi_sentinel.models.policy.transformer_baseline import (
            TransformerActorCritic,
            TransformerSequenceEncoder,
        )
        seq_enc = TransformerSequenceEncoder(
            d_model=seq_cfg.get("d_model", 192),
            n_layers=seq_cfg.get("n_layers", 3),
            n_heads=seq_cfg.get("n_heads", 4),
            max_seq_len=cfg.get("env", {}).get("history_length", 256),
        )
        return TransformerActorCritic(image_encoder=image_enc, sequence_encoder=seq_enc, hidden=trunk_hidden)

    # Default: Mamba — O(L) sequence encoder.
    from aoi_sentinel.models.policy.actor_critic import MambaActorCritic
    from aoi_sentinel.models.vmamba import SequenceEncoder

    seq_enc = SequenceEncoder(
        d_model=seq_cfg.get("d_model", 192),
        n_layers=seq_cfg.get("n_layers", 3),
    )
    return MambaActorCritic(image_encoder=image_enc, sequence_encoder=seq_enc, hidden=trunk_hidden)


def _build_trainer(model, env, cfg: dict):
    from aoi_sentinel.models.policy.lagrangian_ppo import LagrangianPPO, PPOConfig

    ppo_cfg_dict = cfg.get("ppo", {})
    ppo_cfg = PPOConfig(rollout_steps=cfg.get("rollout_steps", 256), **ppo_cfg_dict)
    return LagrangianPPO(model, env, cfg=ppo_cfg)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_dataset(cfg: dict):
    """Load benchmark dataset, fall back to small synthetic if loaders aren't ready.

    Karpathy: get the loop running first; benchmark data plumbing can come
    after the policy is at least training without crashing.
    """
    import numpy as np

    data_cfg = cfg.get("data", {})
    name = data_cfg.get("dataset", "synthetic")
    if name == "synthetic":
        return _synthetic_dataset(cfg)

    try:
        if name == "visa":
            from aoi_sentinel.data.benchmarks.visa import load_visa
            images, labels, _ = load_visa(data_cfg["root"], size=data_cfg.get("roi_size", 224))
        elif name == "deeppcb":
            from aoi_sentinel.data.benchmarks.deeppcb import load_deeppcb
            images, labels, _ = load_deeppcb(data_cfg["root"], size=data_cfg.get("roi_size", 224))
        elif name == "soldef":
            from aoi_sentinel.data.benchmarks.soldef import load_soldef
            images, labels, _ = load_soldef(data_cfg["root"], size=data_cfg.get("roi_size", 224))
        else:
            raise ValueError(f"unknown dataset: {name}")
        return images, labels
    except (ImportError, FileNotFoundError) as e:
        print(f"[stage1] dataset {name!r} unavailable ({e}); falling back to synthetic")
        return _synthetic_dataset(cfg)


def _synthetic_dataset(cfg: dict):
    import numpy as np
    rng = np.random.default_rng(cfg.get("seed", 42))
    n = cfg.get("data", {}).get("n_synthetic", 1024)
    images = rng.integers(0, 255, size=(n, 224, 224, 3), dtype=np.uint8)
    labels = rng.integers(0, 2, size=(n,), dtype=np.int64)
    return images, labels


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def _save_ckpt(model, path: Path, cfg: dict, history: list[dict]) -> None:
    import torch
    torch.save({"state_dict": model.state_dict(), "config": cfg, "history": history}, path)
