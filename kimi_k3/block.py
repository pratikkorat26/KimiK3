"""
kimi_k3/block.py — one Kimi K3 transformer block.

Wires:
  attention   KDA or Gated MLA
  residual    Block AttnRes (depth-wise mix) around attn and MLP
  FFN         StableLatentMoE or DenseFFN (tiny-preset interim path)

Block AttnRes flow (arXiv:2603.15031):
  h = mix(history) → sublayer(PreNorm(h)) → accumulate into history
"""

from __future__ import annotations

import torch.nn as nn

from .attention import GatedMLA, KimiDeltaAttention
from .attention.cache import KDACache, MLACache
from .config import ModelConfig
from .moe import build_ffn
from .norms import rms_norm
from .residuals import BlockAttnRes
from .residuals.depth_history import DepthHistory


class KimiK3Block(nn.Module):
    """Pre-norm block with Block AttnRes around attention and FFN."""

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_type = cfg.attention_type(layer_idx)

        self.attn_norm = rms_norm(cfg.hidden_size, cfg.eps)
        if self.attn_type == "kda":
            self.attn = KimiDeltaAttention(cfg.kda_config())
        else:
            self.attn = GatedMLA(cfg)

        self.mlp_norm = rms_norm(cfg.hidden_size, cfg.eps)
        self.mlp = build_ffn(cfg, layer_idx)

        # Separate pseudo-queries for the two sublayer slots (paper / Moonshot)
        self.attn_res = BlockAttnRes(cfg, layer_idx)
        self.mlp_res = BlockAttnRes(cfg, layer_idx)

    def forward(
        self,
        history: DepthHistory,
        mode: str = "chunk",
        cache: KDACache | MLACache | None = None,
        use_cache: bool = False,
    ) -> tuple[DepthHistory, KDACache | MLACache | None]:
        """Run one layer; updates `history` in place; returns (history, attn_cache)."""
        if not self.attn_res.use_additive:
            history.start_layer(self.layer_idx, self.attn_res.block_size)

        h = self.attn_res.mix(history)                                 # (B, T, d)
        attn_out, new_cache = self.attn(
            self.attn_norm(h), mode=mode, cache=cache, use_cache=use_cache
        )
        self.attn_res.accumulate(history, attn_out)

        h = self.mlp_res.mix(history)
        mlp_out = self.mlp(self.mlp_norm(h))
        self.mlp_res.accumulate(history, mlp_out)

        return history, new_cache
