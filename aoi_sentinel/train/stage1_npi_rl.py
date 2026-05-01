"""Stage 1: online RL on the NPI simulator.

Loads the Phase-0 image encoder weights, wraps a fresh Mamba sequence encoder,
and runs Lagrangian PPO under the escape-rate budget.
"""
from __future__ import annotations

from pathlib import Path

import torch
import yaml

from aoi_sentinel.data.benchmarks import load_deeppcb, load_soldef, load_visa
from aoi_sentinel.models.policy import LagrangianPPO, MambaActorCritic, PPOConfig
from aoi_sentinel.models.vmamba import SequenceEncoder, build_image_encoder
from aoi_sentinel.sim import CostMatrix, NpiEnv

LOADERS = {"visa": load_visa, "deeppcb": load_deeppcb, "soldef": load_soldef}


def run(config_path: str | Path) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    loader = LOADERS[cfg["data"]["dataset"]]
    images, labels, saki_calls = loader(cfg["data"]["root"], size=cfg["data"]["roi_size"])

    cost = CostMatrix(**cfg.get("cost", {}))
    env = NpiEnv(
        images=images,
        labels=labels,
        saki_calls=saki_calls,
        history_length=cfg["env"]["history_length"],
        cost=cost,
        episode_length=cfg["env"]["episode_length"],
        seed=cfg["seed"],
    )

    image_encoder = build_image_encoder(cfg["model"]["image"])
    if cfg["model"].get("init_from"):
        sd = torch.load(cfg["model"]["init_from"], map_location="cpu")
        image_encoder.load_state_dict(sd["encoder"], strict=False)
        print(f"loaded image encoder from {cfg['model']['init_from']}")

    sequence_encoder = SequenceEncoder(
        d_model=cfg["model"]["sequence"]["d_model"],
        n_layers=cfg["model"]["sequence"]["n_layers"],
        max_image_idx=len(images),
    )
    model = MambaActorCritic(image_encoder, sequence_encoder, n_actions=3, hidden=cfg["model"]["trunk_hidden"])

    ppo_cfg = PPOConfig(**cfg["ppo"])
    trainer = LagrangianPPO(model, ppo_cfg, device="cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for it in range(cfg["iterations"]):
        buf = trainer.collect_rollout(env, steps=cfg["rollout_steps"])
        stats = trainer.update(buf)
        print(
            f"iter {it:04d} | actor={stats['actor_loss']:.3f} "
            f"V={stats['value_loss']:.3f} Vc={stats['cost_value_loss']:.3f} "
            f"H={stats['entropy']:.3f} λ={stats['lambda']:.3f} "
            f"avg_cost={stats['avg_cost']:.4f} viol={stats['violation']:+.4f} "
            f"cum_cost={env.cumulative_cost:.1f} cum_escape={env.cumulative_escape}"
        )
        if (it + 1) % cfg["save_every"] == 0:
            torch.save(model.state_dict(), out_dir / f"stage1_iter{it:05d}.pt")
