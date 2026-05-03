"""Actor-critic with two value heads.

    state s = [φ_img(I), φ_seq(history)_last_token]
    π(a|s)   — categorical over {DEFECT, PASS, ESCALATE}
    V(s)     — reward critic (expected return)
    V_c(s)   — cost critic   (expected escape count) — used by Lagrangian PPO
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from aoi_sentinel.models.vmamba import ImageEncoder, SequenceEncoder


class MambaActorCritic(nn.Module):
    def __init__(
        self,
        image_encoder: ImageEncoder,
        sequence_encoder: SequenceEncoder,
        n_actions: int = 3,
        hidden: int = 512,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.sequence_encoder = sequence_encoder

        in_dim = image_encoder.embed_dim + sequence_encoder.d_model
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.value = nn.Linear(hidden, 1)
        self.cost_value = nn.Linear(hidden, 1)

        # Initialise the actor's bias so the policy starts NEUTRAL across the
        # three actions, with a slight nudge AWAY from ESCALATE. Without this
        # the actor's random init can land in the always-ESCALATE basin
        # (a safe but trivial local minimum that satisfies the cost
        # constraint without doing any classification). Pushing the
        # ESCALATE logit down at start gives the policy a fighting chance
        # to discover that classification beats blanket escalation.
        # Action layout: 0=DEFECT, 1=PASS, 2=ESCALATE.
        with torch.no_grad():
            self.actor.bias.copy_(torch.tensor([0.5, 0.0, -0.5]))

    def encode(self, image: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        img_feat = self.image_encoder(image)
        seq_feat = self.sequence_encoder(history)[:, -1]
        return self.trunk(torch.cat([img_feat, seq_feat], dim=-1))

    def forward(self, image: torch.Tensor, history: torch.Tensor):
        h = self.encode(image, history)
        logits = self.actor(h)
        return logits, self.value(h).squeeze(-1), self.cost_value(h).squeeze(-1)

    @torch.no_grad()
    def act(self, image: torch.Tensor, history: torch.Tensor, deterministic: bool = False):
        logits, v, vc = self.forward(image, history)
        dist = Categorical(logits=logits)
        action = dist.probs.argmax(-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, v, vc
