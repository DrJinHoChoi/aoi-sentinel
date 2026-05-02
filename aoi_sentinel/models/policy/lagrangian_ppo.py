"""Lagrangian PPO trainer for the safety-constrained NPI policy.

Solves
        max_π  E[Σ r_t]      s.t.   E[Σ c_t] ≤ ε

via the Lagrangian
        L(π, λ) = E[Σ r_t]  −  λ · ( E[Σ c_t] − ε ),    λ ≥ 0

with PPO's clipped surrogate for the policy update and a separate cost
critic V_c so the safety advantage gets its own GAE. λ is updated by
gradient ascent on the constraint violation; this auto-tunes the
strictness of the safety constraint without manual schedules.

The whole trainer is ~300 lines, no external constrained-RL library —
intentionally lightweight so it runs on a free Colab T4 in minutes per
iteration. All torch imports are lazy so the module can be imported in
torch-free environments (CI lint, Windows without GPU build).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PPOConfig:
    # Optimisation
    n_epochs: int = 4
    minibatch_size: int = 32
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    cost_value_coef: float = 0.5
    max_grad_norm: float = 0.5
    actor_lr: float = 3.0e-4
    critic_lr: float = 1.0e-3

    # GAE / discount
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Lagrangian dual ascent — escape rate budget
    cost_limit: float = 0.001       # ε  — max average escape rate
    init_lambda: float = 1.0
    lambda_lr: float = 0.05
    lambda_max: float = 1000.0      # safety net so λ doesn't blow up on early rollouts

    # Rollout
    rollout_steps: int = 256

    # ImageNet normalisation
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std:  tuple[float, float, float] = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# λ helpers — pure numpy so they're testable without torch
# ---------------------------------------------------------------------------


def softplus_lambda(raw: float) -> float:
    """Numerically-stable softplus, used as the link function for λ."""
    return float(np.log1p(np.exp(-abs(raw))) + max(raw, 0.0))


def update_lambda(
    raw: float,
    mean_cost: float,
    cost_limit: float,
    lr: float,
    lambda_max: float = 1000.0,
) -> float:
    """One step of dual ascent on the unconstrained logit."""
    violation = mean_cost - cost_limit
    new_raw = raw + lr * violation
    # Cap in logit space so the visible λ stays ≤ lambda_max. softplus(x) ≈ x
    # for x > ~20, so beyond that we can use lambda_max directly without
    # overflowing exp() — np.log(np.expm1(1000)) overflows.
    if lambda_max > 20.0:
        max_logit = float(lambda_max)
    else:
        max_logit = float(np.log(np.expm1(lambda_max)))
    return float(min(new_raw, max_logit))


# ---------------------------------------------------------------------------
# Trainer (torch lazily imported inside methods)
# ---------------------------------------------------------------------------


class LagrangianPPO:
    """PPO with a Lagrangian dual variable on the safety constraint.

    The model is any nn.Module returning ``(logits, value, cost_value)`` from
    ``(image, history)`` — see :class:`MambaActorCritic`. Plug in the
    transformer baseline by swapping the model.
    """

    def __init__(self, model, env, cfg: PPOConfig | None = None) -> None:
        import torch

        self.cfg = cfg or PPOConfig()
        self.model = model
        self.env = env
        self.device = next(model.parameters()).device

        # Group params so we can give the actor a smaller LR than the critics.
        actor_params, critic_params = [], []
        for name, p in model.named_parameters():
            (critic_params if "value" in name else actor_params).append(p)
        self.actor_params = actor_params
        self.critic_params = critic_params

        self.optim_actor = torch.optim.AdamW(actor_params, lr=self.cfg.actor_lr)
        self.optim_critic = torch.optim.AdamW(critic_params, lr=self.cfg.critic_lr)

        # λ as a plain float (logit space) — we own the update entirely.
        self._lambda_raw: float = float(np.log(np.expm1(max(self.cfg.init_lambda, 1e-3))))

    # ------------------------------------------------------------------ getters

    @property
    def lagrange_multiplier(self) -> float:
        return softplus_lambda(self._lambda_raw)

    # ------------------------------------------------------------------ rollout

    def collect_rollout(self):
        """Run the env for `rollout_steps` steps with the current policy."""
        from aoi_sentinel.models.policy.rollout_buffer import RolloutBuffer

        cfg = self.cfg
        obs, _ = self.env.reset()
        h, w, c = obs["image"].shape
        buf = RolloutBuffer(
            capacity=cfg.rollout_steps,
            image_shape=(h, w, c),
            history_length=obs["history"].shape[0],
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
        )

        ep_escapes = 0; ep_fc = 0; ep_esc = 0
        for _ in range(cfg.rollout_steps):
            action, log_prob, value, cost_value = self._act(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(int(action))
            done = terminated or truncated
            cost = float(info.get("is_escape", 0))

            buf.add(
                image=obs["image"], history=obs["history"],
                action=int(action), log_prob=float(log_prob),
                value=float(value), cost_value=float(cost_value),
                reward=float(reward), cost=cost, done=done,
            )

            ep_escapes += int(info.get("is_escape", 0))
            if info.get("action") == 2:
                ep_esc += 1
            if info.get("label") == 0 and info.get("action") == 0:
                ep_fc += 1

            obs = next_obs
            if done:
                obs, _ = self.env.reset()

        # Bootstrap final values for GAE.
        _, _, last_value, last_cv = self._act(obs)
        buf.set_terminal_values(float(last_value), float(last_cv))
        buf.compute_advantages()

        stats = {
            "ep_escapes": ep_escapes,
            "ep_false_calls": ep_fc,
            "ep_escalations": ep_esc,
            "mean_cost_return": float(buf.cost_returns[: len(buf)].mean()),
            "mean_reward": float(buf.rewards[: len(buf)].mean()),
        }
        return buf, stats

    # ------------------------------------------------------------------ update

    def update(self, buf) -> dict:
        """One PPO update over the rollout buffer."""
        import torch
        import torch.nn.functional as F

        cfg = self.cfg
        data = buf.to_torch(self.device)

        mean = torch.tensor(cfg.image_mean, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(cfg.image_std, device=self.device).view(1, 3, 1, 1)

        n = data["actions"].size(0)
        idxs = np.arange(n)

        lam = self.lagrange_multiplier
        log = {"policy_loss": 0.0, "value_loss": 0.0, "cost_value_loss": 0.0,
               "entropy": 0.0, "kl": 0.0, "clip_frac": 0.0, "lambda": lam}
        n_minibatches = 0

        for _ in range(cfg.n_epochs):
            np.random.shuffle(idxs)
            for start in range(0, n, cfg.minibatch_size):
                mb = idxs[start : start + cfg.minibatch_size]
                if len(mb) == 0:
                    continue
                mb_t = torch.as_tensor(mb, device=self.device, dtype=torch.long)

                images = data["images"][mb_t].permute(0, 3, 1, 2).float() / 255.0
                images = (images - mean) / std
                histories = data["histories"][mb_t]

                logits, value, cost_value = self.model(images, histories)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(data["actions"][mb_t])
                entropy = dist.entropy().mean()

                # Combined advantage: A − λ · A_c.  Pure PPO clipped surrogate.
                adv = data["advantages"][mb_t] - lam * data["cost_advantages"][mb_t]

                ratio = torch.exp(new_log_probs - data["log_probs"][mb_t])
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(value, data["returns"][mb_t])
                cost_value_loss = F.mse_loss(cost_value, data["cost_returns"][mb_t])

                loss = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    + cfg.cost_value_coef * cost_value_loss
                    - cfg.entropy_coef * entropy
                )

                self.optim_actor.zero_grad(); self.optim_critic.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor_params) + list(self.critic_params),
                    cfg.max_grad_norm,
                )
                self.optim_actor.step(); self.optim_critic.step()

                with torch.no_grad():
                    kl = (data["log_probs"][mb_t] - new_log_probs).mean().abs()
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip_eps).float().mean()

                log["policy_loss"]     += policy_loss.item()
                log["value_loss"]      += value_loss.item()
                log["cost_value_loss"] += cost_value_loss.item()
                log["entropy"]         += entropy.item()
                log["kl"]              += float(kl)
                log["clip_frac"]       += float(clip_frac)
                n_minibatches += 1

        if n_minibatches:
            for k in ("policy_loss", "value_loss", "cost_value_loss", "entropy", "kl", "clip_frac"):
                log[k] /= n_minibatches
        return log

    # ------------------------------------------------------------------ one step

    def step(self) -> dict:
        """Collect a rollout, run one PPO update, dual-ascend λ."""
        buf, rollout_stats = self.collect_rollout()
        update_log = self.update(buf)
        # λ update happens AFTER PPO so the next rollout sees the adjusted λ.
        self._lambda_raw = update_lambda(
            self._lambda_raw,
            mean_cost=rollout_stats["mean_cost_return"],
            cost_limit=self.cfg.cost_limit,
            lr=self.cfg.lambda_lr,
            lambda_max=self.cfg.lambda_max,
        )
        return {**rollout_stats, **update_log, "lambda": self.lagrange_multiplier}

    # ------------------------------------------------------------------ inference

    def _act(self, obs):
        import torch
        cfg = self.cfg

        image = (
            torch.from_numpy(obs["image"]).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
        )
        mean = torch.tensor(cfg.image_mean, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(cfg.image_std, device=self.device).view(1, 3, 1, 1)
        image = (image - mean) / std
        history = torch.from_numpy(obs["history"]).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, value, cost_value = self.model(image, history)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item()), float(cost_value.item())
