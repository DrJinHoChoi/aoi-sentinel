"""Mamba sequence encoder over inspection history.

Each step in `NpiEnv` appends a 5-dim history vector:
    [image_idx, saki_call, action_taken, label_revealed_mask, label_value]

We tokenize that into a learned embedding, optionally fuse the cached image
feature for that step, and run a stack of Mamba blocks. Linear in L.

Output: per-step hidden states (B, L, D). Trainer typically reads the final
position as the policy context.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _MambaBlock(nn.Module):
    """Mamba block wrapper. Prefers the mamba_ssm CUDA kernel; transparently
    falls back to a pure-PyTorch implementation when mamba_ssm is unavailable
    (e.g. Colab with a torch version that has no prebuilt wheel)."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        from aoi_sentinel.models.vmamba.pure_torch_mamba import get_mamba_block

        self.mamba = get_mamba_block(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mamba(self.norm(x))


class SequenceEncoder(nn.Module):
    """Mamba-SSM over the inspection-history stream."""

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        max_image_idx: int = 200_000,
        n_actions: int = 3,
        n_labels: int = 2,
        d_state: int = 16,
        image_feat_dim: int | None = None,
    ) -> None:
        super().__init__()
        # Categorical embeddings — image-idx is just a stable id for the encoder
        # to learn temporal correlations. For inference on unseen lots we drop
        # this embedding (set to a fixed [UNK]) and rely on image_feat fusion.
        self.image_emb = nn.Embedding(max_image_idx + 1, d_model // 4)
        self.saki_emb = nn.Embedding(2, d_model // 8)
        self.action_emb = nn.Embedding(n_actions + 1, d_model // 4)  # +1 for "no action yet"
        self.label_emb = nn.Embedding(n_labels + 1, d_model // 4)    # +1 for masked

        used = (d_model // 4) + (d_model // 8) + (d_model // 4) + (d_model // 4)
        # Optional image-feature fusion (rich token).
        self.image_feat_proj = (
            nn.Linear(image_feat_dim, d_model - used) if image_feat_dim else None
        )
        if self.image_feat_proj is None:
            # Pad to d_model
            self.token_pad = nn.Linear(used, d_model)
        else:
            self.token_pad = None

        self.blocks = nn.ModuleList(
            [_MambaBlock(d_model=d_model, d_state=d_state) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(
        self,
        history: torch.Tensor,             # (B, L, 5) float
        image_feats: torch.Tensor | None = None,  # (B, L, image_feat_dim) optional
    ) -> torch.Tensor:
        b, lseq, _ = history.shape

        image_idx = history[..., 0].clamp(min=0).long()
        saki = history[..., 1].clamp(min=0).long()
        action = history[..., 2].clamp(min=0).long()
        label_mask = history[..., 3]
        label_val = history[..., 4].clamp(min=0).long()

        # When mask=0, label is unrevealed → treat as "masked" id (= n_labels)
        label_id = torch.where(label_mask > 0.5, label_val, torch.full_like(label_val, 2))

        tok = torch.cat(
            [
                self.image_emb(image_idx),
                self.saki_emb(saki),
                self.action_emb(action),
                self.label_emb(label_id),
            ],
            dim=-1,
        )

        if self.image_feat_proj is not None and image_feats is not None:
            tok = torch.cat([tok, self.image_feat_proj(image_feats)], dim=-1)
        else:
            tok = self.token_pad(tok)

        for blk in self.blocks:
            tok = blk(tok)
        return self.final_norm(tok)
