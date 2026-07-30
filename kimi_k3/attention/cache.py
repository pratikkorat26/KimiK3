"""
kimi_k3/attention/cache.py — class-based decode caches for KDA and Gated MLA.

Small-scale production style: plain dataclasses + tensors. No paging, no
memory pools. Names describe what is stored (not Moonshot LoRA/RoPE jargon).

Moonshot alias map (comments only — code uses our names):
    kv_latent_dim  ↔  kv_lora_rank
    qk_content_dim ↔  qk_nope_head_dim
    qk_shared_dim  ↔  qk_rope_head_dim   (NoPE: not actually RoPE)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class KDACache:
    """Fixed-size KDA memory (independent of sequence length).

    Shapes:
        recurrent: (B, H, d_k, d_v) — associative state S
        conv_q/k:  (B, conv_kernel-1, H*d_k)
        conv_v:    (B, conv_kernel-1, H*d_v)
    """

    recurrent: Tensor
    conv_q: Tensor
    conv_k: Tensor
    conv_v: Tensor


@dataclass
class MLACache:
    """Growing MLA memory (length = tokens seen so far).

    Stores the compressed representation; up-project when attending.

    Shapes:
        kv_latent: (B, T, kv_latent_dim) — c_kv
        k_shared:  (B, T, qk_shared_dim) — shared key channels (NoPE, not RoPE)
    """

    kv_latent: Tensor
    k_shared: Tensor

    @property
    def seq_len(self) -> int:
        return self.kv_latent.shape[1]

    def append(self, kv_latent: Tensor, k_shared: Tensor) -> MLACache:
        """Concatenate along T; return a new MLACache (autograd-friendly)."""
        return MLACache(
            kv_latent=torch.cat([self.kv_latent, kv_latent], dim=1),
            k_shared=torch.cat([self.k_shared, k_shared], dim=1),
        )


@dataclass
class AttentionCache:
    """Model decode cache: one entry per layer (KDACache | MLACache | None)."""

    layers: list[KDACache | MLACache | None] = field(default_factory=list)
    tokens_seen: int = 0

    @classmethod
    def empty(cls, n_layer: int) -> AttentionCache:
        return cls(layers=[None] * n_layer, tokens_seen=0)

    def get(self, layer_idx: int) -> KDACache | MLACache | None:
        return self.layers[layer_idx]

    def set(self, layer_idx: int, layer_cache: KDACache | MLACache) -> None:
        self.layers[layer_idx] = layer_cache

    def __len__(self) -> int:
        return len(self.layers)
