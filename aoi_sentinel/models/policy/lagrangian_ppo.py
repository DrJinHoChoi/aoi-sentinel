"""Lagrangian PPO trainer.

Constrained MDP:
    max_θ  J(π_θ)            s.t.   J_c(π_θ) ≤ ε
    L(θ, λ) = J(π_θ) - λ (J_c(π_θ) - ε),   λ ≥ 0

Implementation:
    - Two value heads (reward + cost)
    - GAE on both reward- and cost-advantages
    - PPO clipped objective on the actor loss = -A_r(s,a) + λ * A_c(s,a)
    - λ is a learnable scalar with softplus parametrization, updated by gradient ascent
      on (J_c - ε) — i.e. λ grows if we are violating the safety budget.

Lightweight (~250 lines) by design — no distributed training, no LR schedules,
no fancy advantage normalization beyond mean/std. Get the math right first;
optimize later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical


@dataclass
class PPOConfig:
    n_epochs: int = 4
    minibatch_size: int = 64
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    cost_value_coef: float = 0.5
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    cost_limit: float = 0.001          # ε — escape rate budget per episode (fraction)
    lambda_lr: float = 0.05            # dual ascent step size
    init_lambda: float = 1.0
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3


class LagrangianPPO:
    def __init__(self, model, config: PPOConfig | None = None, device: str = "cuda"):
        self.model = model.to(device)
        self.cfg = config or PPOConfig()
        self.device = device
        self.optim = torch.optim.AdamW(model.parameters(), lr=self.cfg.actor_lr)
        # Parametrize λ via raw scalar → softplus to keep it ≥ 0.
        init_raw = float(np.log(np.expm1(max(self.cfg.init_lambda, 1e-3))))
        self.lambda_raw = torch.nn.Parameter(torch.tensor(init_raw, device=device))

    @property
    def lagrange_lambda(self) -> float:
        return float(F.softplus(self.lambda_raw).item())

    # ------------------------------------------------------------------ rollout

    def collect_rollout(self, env, steps: int):
        obs, _ = env.reset()
        buf = _RolloutBuffer()
        for _ in range(steps):
            image = self._to_tensor(obs["image"]).unsqueeze(0)
            history = torch.from_numpy(obs["history"]).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, log_prob, v, vc = self.model.act(image, history)
            a = int(action.item())
            next_obs, reward, term, trunc, info = env.step(a)
            buf.add(
                obs=obs,
                action=a,
                reward=float(reward),
                cost=float(info["is_escape"]),
                log_prob=float(log_prob.item()),
                value=float(v.item()),
                cost_value=float(vc.item()),
                done=term or trunc,
            )
            obs = next_obs
            if term or trunc:
                obs, _ = env.reset()
        return buf

    # ------------------------------------------------------------------ update

    def update(self, buf: "_RolloutBuffer"):
        adv_r, ret_r = self._gae(buf.rewards, buf.values, buf.dones)
        adv_c, ret_c = self._gae(buf.costs, buf.cost_values, buf.dones)
        adv_r = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8)

        images = torch.stack([self._to_tensor(o["image"]) for o in buf.obs])
        histories = torch.from_numpy(np.stack([o["history"] for o in buf.obs])).float().to(self.device)
        actions = torch.tensor(buf.actions, device=self.device)
        old_log_probs = torch.tensor(buf.log_probs, device=self.device, dtype=torch.float32)
        ret_r_t = torch.tensor(ret_r, device=self.device, dtype=torch.float32)
        ret_c_t = torch.tensor(ret_c, device=self.device, dtype=torch.float32)
        adv_r_t = torch.tensor(adv_r, device=self.device, dtype=torch.float32)
        adv_c_t = torch.tensor(adv_c, device=self.device, dtype=torch.float32)

        n = len(actions)
        idx = np.arange(n)

        for _ in range(self.cfg.n_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.cfg.minibatch_size):
                mb = idx[start : start + self.cfg.minibatch_size]
                logits, v, vc = self.model(images[mb], histories[mb])
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions[mb])
                entropy = dist.entropy().mean()

                ratio = (new_log_probs - old_log_probs[mb]).exp()
                lam = F.softplus(self.lambda_raw)
                # Combined advantage: maximize reward, minimize cost (weighted by λ).
                adv = adv_r_t[mb] - lam.detach() * adv_c_t[mb]
                clipped = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps)
                actor_loss = -torch.minimum(ratio * adv, clipped * adv).mean()

                v_loss = F.mse_loss(v, ret_r_t[mb])
                vc_loss = F.mse_loss(vc, ret_c_t[mb])

                loss = (
                    actor_loss
                    + self.cfg.value_coef * v_loss
                    + self.cfg.cost_value_coef * vc_loss
                    - self.cfg.entropy_coef * entropy
                )

                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optim.step()

        # Dual ascent on λ — push it up if average cost exceeded the budget.
        avg_cost = float(np.mean(buf.costs))
        violation = avg_cost - self.cfg.cost_limit
        with torch.no_grad():
            self.lambda_raw += self.cfg.lambda_lr * violation
        return {
            "actor_loss": float(actor_loss.item()),
            "value_loss": float(v_loss.item()),
            "cost_value_loss": float(vc_loss.item()),
            "entropy": float(entropy.item()),
            "lambda": self.lagrange_lambda,
            "avg_cost": avg_cost,
            "violation": violation,
        }

    # ------------------------------------------------------------------ helpers

    def _gae(self, rewards, values, dones):
        rewards = np.asarray(rewards, dtype=np.float32)
        values = np.asarray(values + [values[-1]], dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)
        adv = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            nonterm = 1.0 - dones[t]
            delta = rewards[t] + self.cfg.gamma * values[t + 1] * nonterm - values[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * nonterm * gae
            adv[t] = gae
        ret = adv + values[:-1]
        return adv, ret

    def _to_tensor(self, image_np: np.ndarray) -> torch.Tensor:
        # uint8 HWC → float CHW [0, 1] → simple ImageNet-style normalization.
        t = torch.from_numpy(image_np).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return ((t - mean) / std).to(self.device)


@dataclass
class _RolloutBuffer:
    obs: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    costs: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    values: list = field(default_factory=list)
    cost_values: list = field(default_factory=list)
    dones: list = field(default_factory=list)

    def add(self, **kw: Any) -> None:
        self.obs.append(kw["obs"])
        self.actions.append(kw["action"])
        self.rewards.append(kw["reward"])
        self.costs.append(kw["cost"])
        self.log_probs.append(kw["log_prob"])
        self.values.append(kw["value"])
        self.cost_values.append(kw["cost_value"])
        self.dones.append(kw["done"])
