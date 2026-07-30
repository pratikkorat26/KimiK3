"""
kimi_k3/attention/gated_mla.py — Gated Multi-head Latent Attention.

Base structure: Kimi Linear `KimiMLAAttention` (adapted from DeepSeek MLA,
NoPE forced). Kimi K3 uses a query LoRA bottleneck and full-rank output gate.

Naming (ours → Moonshot alias in comments only):
    kv_latent_dim   ↔ kv_lora_rank
    qk_content_dim  ↔ qk_nope_head_dim
    qk_shared_dim   ↔ qk_rope_head_dim   (NoPE: channels exist, RoPE is NOT applied)
    w_kv_down       ↔ kv_a_proj_with_mqa
    w_kv_up         ↔ kv_b_proj

Forward math:
    q = w_q_up(RMSNorm(w_q_down(x))) → (B, H, T, q_head_dim)
    [c_kv | k_shared] = w_kv_down(x)
    k_content, v = split(w_kv_up(kv_norm(c_kv)))
    k = concat(k_content, expand(k_shared))
    o = causal_attn(q, k, v)
    out = w_o( sigmoid(w_gate(x)) ⊙ o )

Cache: MLACache(kv_latent, k_shared) — compressed, grows with T.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .cache import MLACache


class GatedMLA(nn.Module):
    """Gated Multi-head Latent Attention (NoPE) for the Kimi K3 hybrid stack.

    Shapes:
        x:     (B, T, d)
        out:   (B, T, d)
        cache: MLACache | None
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_size
        H = cfg.n_heads
        d_content = cfg.qk_content_dim
        d_shared = cfg.qk_shared_dim
        d_v = cfg.v_head_dim
        d_c = cfg.kv_latent_dim
        q_dim = cfg.q_head_dim                       # content + shared
        q_rank = cfg.q_lora_rank

        self.n_heads = H
        self.qk_content_dim = d_content
        self.qk_shared_dim = d_shared
        self.v_head_dim = d_v
        self.kv_latent_dim = d_c
        self.q_head_dim = q_dim
        self.scale = q_dim ** -0.5

        # --- projections (Kimi Linear structure; clear names) ---
        self.w_q_down = nn.Linear(d, q_rank, bias=False)
        self.q_norm = nn.RMSNorm(q_rank, eps=cfg.eps)
        self.w_q_up = nn.Linear(q_rank, H * q_dim, bias=False)
        self.w_kv_down = nn.Linear(d, d_c + d_shared, bias=False)      # → [c_kv | k_shared]
        self.kv_norm = nn.RMSNorm(d_c, eps=cfg.eps)
        self.w_kv_up = nn.Linear(d_c, H * (d_content + d_v), bias=False)

        # K3 uses a full-rank output gate for MLA.
        self.w_gate = nn.Linear(d, H * d_v, bias=False)
        self.w_o = nn.Linear(H * d_v, d, bias=False)

    def _up_kv(
        self, kv_latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """kv_latent (B, T, d_c) → k_content (B,H,T,d_content), v (B,H,T,d_v)."""
        B, T, _ = kv_latent.shape
        H, d_c, d_v = self.n_heads, self.qk_content_dim, self.v_head_dim
        up = self.w_kv_up(self.kv_norm(kv_latent))                     # (B, T, H*(d_c+d_v))
        up = up.view(B, T, H, d_c + d_v).transpose(1, 2)               # (B, H, T, d_c+d_v)
        k_content, v = up.split([d_c, d_v], dim=-1)
        return k_content, v

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        prefix_len: int = 0,
    ) -> torch.Tensor:
        """Causal (or decode) attention.

        q: (B, H, T_q, q_head_dim); k: (B, H, T_k, q_head_dim); v: (B, H, T_k, d_v)
        """
        T_q, T_k = q.shape[2], k.shape[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale     # (B, H, T_q, T_k)
        query_positions = prefix_len + torch.arange(T_q, device=q.device)
        key_positions = torch.arange(T_k, device=q.device)
        causal = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        scores = scores.masked_fill(causal, float("-inf"))
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        return torch.matmul(attn, v)                                   # (B, H, T_q, d_v)

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "chunk",
        cache: MLACache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, MLACache | None]:
        """Run Gated MLA. `mode` is accepted for API parity with KDA (ignored)."""
        del mode
        B, T, _ = x.shape
        H = self.n_heads
        d_content, d_shared, d_v = self.qk_content_dim, self.qk_shared_dim, self.v_head_dim
        d_c = self.kv_latent_dim
        if cache is not None:
            if not isinstance(cache, MLACache):
                raise TypeError(
                    f"cache must be MLACache or None, got {type(cache).__name__}"
                )
            if cache.kv_latent.ndim != 3 or cache.k_shared.ndim != 3:
                raise ValueError("MLA cache tensors must both have rank 3")
            if cache.kv_latent.shape[:2] != cache.k_shared.shape[:2]:
                raise ValueError("MLA cache tensors must share batch and sequence dimensions")
            if cache.kv_latent.shape[0] != B:
                raise ValueError(
                    f"MLA cache batch size must be {B}, got {cache.kv_latent.shape[0]}"
                )
            if cache.kv_latent.shape[-1] != d_c:
                raise ValueError(
                    f"MLA kv_latent width must be {d_c}, got {cache.kv_latent.shape[-1]}"
                )
            if cache.k_shared.shape[-1] != d_shared:
                raise ValueError(
                    f"MLA k_shared width must be {d_shared}, got {cache.k_shared.shape[-1]}"
                )
            if cache.kv_latent.device != cache.k_shared.device:
                raise ValueError("MLA cache tensors must share a device")
            if cache.kv_latent.dtype != cache.k_shared.dtype:
                raise TypeError("MLA cache tensors must share a dtype")

        # --- 1. queries: low-rank bottleneck, then split content | shared ---
        q = self.w_q_up(self.q_norm(self.w_q_down(x)))
        q = q.view(B, T, H, self.q_head_dim).transpose(1, 2)
        # (kept as one tensor; split only if inspecting — concat order is content|shared)

        # --- 2. KV down-project → latent + shared key channels -------------
        kv_down = self.w_kv_down(x)                                    # (B, T, d_c+d_shared)
        kv_latent_new, k_shared_new = kv_down.split([d_c, d_shared], dim=-1)
        # kv_latent_new: (B, T, d_c);  k_shared_new: (B, T, d_shared)

        prefix_len = 0
        if cache is not None:
            if cache.kv_latent.device != kv_latent_new.device:
                raise ValueError(
                    "MLA cache device does not match the current input "
                    f"({cache.kv_latent.device} != {kv_latent_new.device})"
                )
            if cache.kv_latent.dtype != kv_latent_new.dtype:
                raise TypeError(
                    "MLA cache dtype does not match the current projections "
                    f"({cache.kv_latent.dtype} != {kv_latent_new.dtype})"
                )
            prefix_len = cache.seq_len
            full = cache.append(kv_latent_new, k_shared_new)
            kv_latent, k_shared = full.kv_latent, full.k_shared
        else:
            kv_latent, k_shared = kv_latent_new, k_shared_new
            full = MLACache(kv_latent=kv_latent, k_shared=k_shared)

        # --- 3. up-project latent; expand shared keys to all heads ---------
        k_content, v = self._up_kv(kv_latent)                          # (B, H, T_kv, ·)
        k_shared_h = k_shared.unsqueeze(1).expand(-1, H, -1, -1)       # (B, H, T_kv, d_shared)
        k = torch.cat([k_content, k_shared_h], dim=-1)                 # (B, H, T_kv, q_dim)

        # --- 4. attention + K3 gate + output --------------------------------
        o = self._attend(q, k, v, prefix_len=prefix_len)               # (B, H, T, d_v)
        o = o.transpose(1, 2).reshape(B, T, H * d_v)                   # (B, T, H*d_v)
        gate = torch.sigmoid(self.w_gate(x))                           # (B, T, H*d_v)
        out = self.w_o(o * gate)                                       # (B, T, d)

        return out, (full if use_cache else None)
