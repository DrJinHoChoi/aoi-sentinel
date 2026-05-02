"""Torch implementation of the transformer baseline (separated for lazy import)."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical


class _TransformerSeqEncImpl(nn.Module):
    """Standard pre-norm Transformer encoder over the (L, 5) history stream.

    Notable for benchmarking: causal mask makes self-attention O(L²) — the
    headline complexity we are comparing Mamba against. Token construction
    matches the Mamba sequence encoder so the comparison is apples-to-apples.
    """

    def __init__(
        self,
        d_model: int = 192,
        n_layers: int = 3,
        n_heads: int = 4,
        max_image_idx: int = 200_000,
        n_actions: int = 3,
        n_labels: int = 2,
        max_seq_len: int = 4096,
    ) -> None:
        super().__init__()
        self.image_emb = nn.Embedding(max_image_idx + 1, d_model // 4)
        self.saki_emb = nn.Embedding(2, d_model // 8)
        self.action_emb = nn.Embedding(n_actions + 1, d_model // 4)
        self.label_emb = nn.Embedding(n_labels + 1, d_model // 4)
        used = (d_model // 4) + (d_model // 8) + (d_model // 4) + (d_model // 4)
        self.token_pad = nn.Linear(used, d_model)

        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        b, lseq, _ = history.shape
        image_idx = history[..., 0].clamp(min=0).long()
        saki = history[..., 1].clamp(min=0).long()
        action = history[..., 2].clamp(min=0).long()
        label_mask = history[..., 3]
        label_val = history[..., 4].clamp(min=0).long()
        label_id = torch.where(label_mask > 0.5, label_val, torch.full_like(label_val, 2))

        tok = torch.cat(
            [self.image_emb(image_idx), self.saki_emb(saki),
             self.action_emb(action), self.label_emb(label_id)],
            dim=-1,
        )
        tok = self.token_pad(tok)
        pos = self.pos_emb(torch.arange(lseq, device=tok.device))
        tok = tok + pos.unsqueeze(0)

        # Causal mask — same temporal semantics as Mamba's selective scan.
        mask = torch.triu(torch.full((lseq, lseq), float("-inf"), device=tok.device), diagonal=1)
        tok = self.encoder(tok, mask=mask)
        return self.final_norm(tok)


class _TransformerActorCriticImpl(nn.Module):
    """Same actor-critic shape as :class:`MambaActorCritic` with a transformer
    history encoder. Vision encoder is held externally and passed in."""

    def __init__(
        self,
        image_encoder,
        sequence_encoder,
        n_actions: int = 3,
        hidden: int = 384,
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

    def forward(self, image: torch.Tensor, history: torch.Tensor):
        img_feat = self.image_encoder(image)
        seq_feat = self.sequence_encoder(history)[:, -1]
        h = self.trunk(torch.cat([img_feat, seq_feat], dim=-1))
        logits = self.actor(h)
        return logits, self.value(h).squeeze(-1), self.cost_value(h).squeeze(-1)

    @torch.no_grad()
    def act(self, image: torch.Tensor, history: torch.Tensor, deterministic: bool = False):
        logits, v, vc = self.forward(image, history)
        dist = Categorical(logits=logits)
        action = dist.probs.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), v, vc
