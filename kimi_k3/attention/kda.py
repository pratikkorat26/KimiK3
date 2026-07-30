"""
kimi_k3/attention/kda.py — KimiDeltaAttention: the full KDA layer.

Paper: "Kimi Linear: An Expressive, Efficient Attention Architecture"
       (arXiv:2510.26692), Section 3.3 "Neural Parameterization" (Eq. 10)
       and Figure 3. This is the linear-attention backbone of Kimi K3.

Per head h, the layer computes (paper notation):

    q_t, k_t = L2Norm(Swish(ShortConv(W_{q/k} x_t)))     in R^{d_k}
    v_t      =        Swish(ShortConv(W_v    x_t))       in R^{d_v}
    alpha_t  = exp( g_min * Sigmoid(exp(A_h) * z_t) )    in [exp(g_min), 1]^{d_k}
    beta_t   = Sigmoid(W_beta x_t)                       in [0, 1]
    o_t      = W_o [ Sigmoid(W_g x_t)                    (full-rank gate, K3 Eq. 6)
                     ⊙ RMSNorm( KDA(q_t, k_t, v_t, alpha_t, beta_t) ) ]

where z_t = W_up_alpha W_down_alpha x_t + dt_bias are the per-channel decay logits.
Kimi K3 (Eq. 5) uses the lower-bounded SCALED-SIGMOID decay g = g_min·Sigmoid(exp(A_h)·z),
g_min = -5 — NOT the negative-softplus form of Kimi Linear (arXiv:2510.26692). Likewise
K3 uses a full-rank output gate W_g (Eq. 6), where Kimi Linear used a low-rank one.

Notes for studying:
  - alpha is PER-CHANNEL (one decay rate per d_k channel) — this is the
    fine-grained gating that distinguishes KDA from Gated DeltaNet's
    per-head scalar decay.
  - The KDA core maintains a fixed-size recurrent tensor S of shape
    (d_k, d_v) per head (see KDACache), instead of a growing KV cache.
  - In Kimi Linear / Kimi K3, KDA layers are interleaved with Gated MLA
    at a 3:1 ratio. This file is the readable reference; production uses
    the Triton kernel in FLA (fla/ops/kda).

Shape walkthrough of forward():
    x:          (B, T, d)
    q, k:       (B, H, T, d_k)
    v:          (B, H, T, d_v)
    g:          (B, H, T, d_k)       g = log(alpha)
    beta:       (B, H, T)
    core out:   (B, H, T, d_v)
    out:        (B, T, d)
    cache:      KDACache (recurrent + conv windows)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import KDAConfig
from .cache import KDACache
from .kda_ops import kda_chunkwise, kda_recurrence


class KimiDeltaAttention(nn.Module):
    """One Kimi Delta Attention layer (see module docstring for the math).

    Shapes:
        x:     (B, T, d)
        out:   (B, T, d)
        cache: KDACache | None
    """

    def __init__(self, cfg: KDAConfig):
        super().__init__()
        self.cfg = cfg
        d, H = cfg.hidden_size, cfg.num_heads
        d_k, d_v = cfg.head_dim_k, cfg.head_dim_v
        r, K = cfg.gate_rank, cfg.conv_kernel_size

        self.w_q = nn.Linear(d, H * d_k, bias=False)
        self.w_k = nn.Linear(d, H * d_k, bias=False)
        self.w_v = nn.Linear(d, H * d_v, bias=False)

        self.conv_q = nn.Conv1d(H * d_k, H * d_k, K, groups=H * d_k)
        self.conv_k = nn.Conv1d(H * d_k, H * d_k, K, groups=H * d_k)
        self.conv_v = nn.Conv1d(H * d_v, H * d_v, K, groups=H * d_v)

        self.w_alpha_down = nn.Linear(d, r, bias=False)
        self.w_alpha_up = nn.Linear(r, H * d_k, bias=False)
        # Kimi K3 (Eq. 5) initializes the per-head log-scale A_h = 0 (so exp(A_h) = 1),
        # unlike the Mamba S4D range init. dt_bias is the per-channel decay bias b_alpha.
        self.A_log = nn.Parameter(torch.zeros(H, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.zeros(H * d_k, dtype=torch.float32))

        self.w_beta = nn.Linear(d, H, bias=False)

        self.norm = nn.RMSNorm(d_v, eps=cfg.eps)
        if cfg.use_full_rank_gate:
            self.w_gate = nn.Linear(d, H * d_v, bias=False)
        else:
            self.w_gate_down = nn.Linear(d, r, bias=False)
            self.w_gate_up = nn.Linear(r, H * d_v, bias=False)
        self.w_o = nn.Linear(H * d_v, d, bias=False)

        self.q_scale = d_k ** -0.5 if cfg.use_q_scale else 1.0

    def _short_conv(
        self,
        x_flat: torch.Tensor,
        conv: nn.Conv1d,
        conv_window: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Causal depthwise ShortConv + Swish on one q/k/v branch.

        Shapes:
            x_flat:      (B, T, H*d_branch)
            conv_window: (B, K-1, H*d_branch) or None
        Returns:
            y:           (B, T, H*d_branch)
            new_window:  (B, K-1, H*d_branch)
        """
        K = self.cfg.conv_kernel_size
        if conv_window is not None:
            if conv_window.device != x_flat.device:
                raise ValueError(
                    "KDA convolution cache device does not match the current input "
                    f"({conv_window.device} != {x_flat.device})"
                )
            if conv_window.dtype != x_flat.dtype:
                raise TypeError(
                    "KDA convolution cache dtype does not match the current projections "
                    f"({conv_window.dtype} != {x_flat.dtype})"
                )
            x_full = torch.cat([conv_window, x_flat], dim=1)
        else:
            x_full = F.pad(x_flat, (0, 0, K - 1, 0))
        new_window = x_full[:, -(K - 1):, :] if K > 1 else x_full[:, :0, :]
        y = conv(x_full.transpose(1, 2)).transpose(1, 2)
        return F.silu(y), new_window

    def _decay_gate(self, x: torch.Tensor) -> torch.Tensor:
        """Per-channel log-decay g = log(alpha) — Kimi K3's lower-bounded scaled sigmoid.

        Kimi K3 (report arXiv:2607.24653, Eq. 5) DELIBERATELY REPLACES Kimi Linear's
        negative-softplus decay with a sigmoid bounded from below:

            z = W_up_alpha W_down_alpha x + dt_bias       (per-channel decay logits)
            g = g_min * Sigmoid(exp(A_h) * z)  in (g_min, 0)^{d_k}
            alpha = exp(g)                     in (exp(g_min), 1)^{d_k}

        g_min = gate_lower_bound = -5 (fixed); A_h is a learnable per-head log-scale
        (init 0). The finite range keeps the cumulative chunk decay within the bf16
        dynamic range, letting the chunkwise kernel use dense Tensor-Core tiles.
        Do NOT switch this to the softplus form — that is Kimi Linear, not K3.
        Computed in fp32.
        """
        B, T, _ = x.shape
        H, d_k = self.cfg.num_heads, self.cfg.head_dim_k
        z = self.w_alpha_up(self.w_alpha_down(x)).view(B, T, H, d_k)
        dt_bias = self.dt_bias.view(H, d_k)
        return self.cfg.gate_lower_bound * torch.sigmoid(
            self.A_log.exp().view(1, 1, H, 1) * (z.float() + dt_bias)
        )

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "chunk",
        cache: KDACache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KDACache | None]:
        """Run the KDA layer.

        mode:  "chunk" for training/prefill, "recurrent" for decode.
        cache: optional KDACache from a previous call.
        """
        B, T, _ = x.shape
        cfg = self.cfg
        H, d_k, d_v = cfg.num_heads, cfg.head_dim_k, cfg.head_dim_v
        if cache is not None:
            if not isinstance(cache, KDACache):
                raise TypeError(
                    f"cache must be KDACache or None, got {type(cache).__name__}"
                )
            expected = {
                "recurrent": (B, H, d_k, d_v),
                "conv_q": (B, cfg.conv_kernel_size - 1, H * d_k),
                "conv_k": (B, cfg.conv_kernel_size - 1, H * d_k),
                "conv_v": (B, cfg.conv_kernel_size - 1, H * d_v),
            }
            for name, shape in expected.items():
                value = getattr(cache, name)
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"KDA cache {name} must be a torch.Tensor")
                if tuple(value.shape) != shape:
                    raise ValueError(
                        f"KDA cache {name} must have shape {shape}, got {tuple(value.shape)}"
                    )
                if value.device != x.device:
                    raise ValueError(
                        f"KDA cache {name} device does not match input "
                        f"({value.device} != {x.device})"
                    )
            conv_dtypes = {cache.conv_q.dtype, cache.conv_k.dtype, cache.conv_v.dtype}
            if len(conv_dtypes) != 1:
                raise TypeError("KDA convolution cache tensors must share a dtype")

        conv_q = cache.conv_q if cache is not None else None
        conv_k = cache.conv_k if cache is not None else None
        conv_v = cache.conv_v if cache is not None else None
        S_0 = cache.recurrent if cache is not None else None

        q, cs_q = self._short_conv(self.w_q(x), self.conv_q, conv_q)
        k, cs_k = self._short_conv(self.w_k(x), self.conv_k, conv_k)
        v, cs_v = self._short_conv(self.w_v(x), self.conv_v, conv_v)

        q = F.normalize(q.view(B, T, H, d_k), dim=-1, eps=cfg.eps) * self.q_scale
        k = F.normalize(k.view(B, T, H, d_k), dim=-1, eps=cfg.eps)
        v = v.view(B, T, H, d_v)

        g = self._decay_gate(x)
        beta = torch.sigmoid(self.w_beta(x))

        q, k, v, g, beta = (t.transpose(1, 2) for t in (q, k, v, g, beta))
        if mode == "chunk":
            o, S = kda_chunkwise(q, k, v, g, beta, cfg.chunk_size, S_0)
        elif mode == "recurrent":
            o, S = kda_recurrence(q, k, v, g, beta, S_0)
        else:
            raise ValueError(f"unknown mode: {mode!r} (expected 'chunk' or 'recurrent')")

        o = self.norm(o.transpose(1, 2))
        o = o.reshape(B, T, H * d_v)
        if cfg.use_full_rank_gate:
            gate = torch.sigmoid(self.w_gate(x))
        else:
            gate = torch.sigmoid(self.w_gate_up(self.w_gate_down(x)))
        out = self.w_o(o * gate)

        new_cache = None
        if use_cache:
            new_cache = KDACache(recurrent=S, conv_q=cs_q, conv_k=cs_k, conv_v=cs_v)
        return out, new_cache
