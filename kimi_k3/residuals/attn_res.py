"""
kimi_k3/residuals/attn_res.py — Block Attention Residuals.

Paper: "Attention Residuals" (arXiv:2603.15031), Block AttnRes variant.

Replaces fixed residual accumulation x ← x + f(x) with depth-wise softmax
attention over block-level representations:

    h = sum_i  alpha_i * V_i
    alpha = softmax_i( w^T RMSNorm(V_i) )

where w is a learned per-sublayer pseudo-query, and
V = completed blocks ∪ {partial}. Scoring and mixing are computed in fp32.

When use_interim_residual=True, falls back to additive residuals (ablation /
legacy) and DepthHistory.partial carries the usual residual stream.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .depth_history import DepthHistory


class BlockAttnRes(nn.Module):
    """One AttnRes mixer (pseudo-query + key RMSNorm) for a single sublayer slot.

    Shapes:
        history.sources(): list of (B, T, d)
        mix() → (B, T, d)  — input to the next PreNorm sublayer
    """

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.block_size = cfg.attn_res_block_size
        self.use_additive = cfg.use_interim_residual

        # Pseudo-query w ∈ R^d stored as Linear(d, 1). Model-level
        # initialization gives it the same normal initialization as Moonshot.
        self.query = nn.Linear(cfg.hidden_size, 1, bias=False)
        self.key_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.eps)

    def mix(self, history: DepthHistory) -> torch.Tensor:
        """Depth-wise softmax mix over history sources → next sublayer input."""
        if self.use_additive:
            # Additive path: partial holds the residual stream (or embedding)
            assert history.partial is not None
            return history.partial

        sources = history.sources()
        assert len(sources) >= 1
        if len(sources) == 1:
            return sources[0]  # only embedding (or single block): identity

        # V: (N, B, T, d)
        values = torch.stack(sources, dim=0)
        values_float = values.float()
        variance = values_float.square().mean(dim=-1, keepdim=True)
        keys = values_float * torch.rsqrt(variance + self.cfg.eps)
        w = self.key_norm.weight.float() * self.query.weight.squeeze(0).float()
        logits = torch.einsum("d,nbtd->nbt", w, keys)                  # (N, B, T)
        alpha = F.softmax(logits, dim=0)                               # (N, B, T)
        mixed = torch.einsum("nbt,nbtd->btd", alpha, values_float)
        return mixed.to(values.dtype)                                  # (B, T, d)

    def accumulate(self, history: DepthHistory, sublayer_out: torch.Tensor) -> None:
        """Add a sublayer output to the current decoder-layer block sum."""
        if self.use_additive:
            # Standard residual: stream ← stream + sublayer_out
            assert history.partial is not None
            history.partial = history.partial + sublayer_out
            return

        history.accumulate(sublayer_out)
