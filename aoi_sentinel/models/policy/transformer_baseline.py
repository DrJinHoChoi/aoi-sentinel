"""Transformer-based actor-critic — the comparison baseline.

Karpathy: every claim of "X is better" needs a baseline. We pair every
Mamba sequence encoder with a transformer of comparable parameter count
so the bench script can show the O(N) vs O(N²) advantage empirically.

This module is structurally identical to :class:`MambaActorCritic` but
swaps the sequence encoder. Same `(image, history) → (logits, value,
cost_value)` interface so :class:`LagrangianPPO` works unchanged.
"""
from __future__ import annotations


class TransformerSequenceEncoder:
    """Lazy facade — only realises the torch module on instantiation."""

    def __new__(cls, *args, **kwargs):
        from aoi_sentinel.models.policy._transformer_impl import _TransformerSeqEncImpl
        return _TransformerSeqEncImpl(*args, **kwargs)


class TransformerActorCritic:
    """Drop-in for MambaActorCritic with a transformer history encoder."""

    def __new__(cls, *args, **kwargs):
        from aoi_sentinel.models.policy._transformer_impl import _TransformerActorCriticImpl
        return _TransformerActorCriticImpl(*args, **kwargs)
