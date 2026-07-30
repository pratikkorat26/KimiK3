"""
kimi_k3/moe/router.py — Quantile Balancing router (study-scale).

Kimi K3 Stable LatentMoE (arXiv:2607.24653, §2.3.3 / Eq. 14):
  scores s = sigmoid(x W_router)                (M tokens × E experts)
  route with Top-k on (s + expert_bias), but weight using unbiased s
  after the batch (training only), update bias via one QB step so the next
  batch sees a corrected bias (causal: never route with a self-derived bias)

Study implementation: one alternating QB iteration (not a multi-iter LP solver,
not distributed histogram estimation). Clear names throughout.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig


class QuantileBalancingRouter(nn.Module):
    """Token → expert router with Quantile Balancing bias correction.

    Shapes:
        x:            (B, T, d_model) or (M, d_model)
        topk_idx:     (B, T, K) or (M, K)
        topk_weight:  (B, T, K) or (M, K)  — renormalized positive weights
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.n_experts_per_tok = cfg.n_experts_per_tok
        self.hidden_size = cfg.hidden_size
        self.routed_scaling_factor = cfg.routed_scaling_factor
        self.w_router = nn.Linear(cfg.hidden_size, cfg.n_experts, bias=False)
        # Frozen at inference; updated in-place during training after routing
        self.expert_bias = nn.Parameter(torch.zeros(cfg.n_experts), requires_grad=False)

    def forward(self, x: torch.Tensor):
        """Route tokens; optionally refresh expert_bias for the *next* batch."""
        leading = x.shape[:-1]                                     # (B, T) or (M,)
        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"router input width must be {self.hidden_size}, got {x.shape[-1]}"
            )
        flat = x.reshape(-1, self.hidden_size)                     # (M, d_model)
        M = flat.shape[0]
        K = self.n_experts_per_tok

        logits = F.linear(
            flat.float(),
            self.w_router.weight.float(),
        )
        scores = torch.sigmoid(logits)                             # (M, E)
        biased = scores + self.expert_bias                         # (M, E)
        _, topk_idx = torch.topk(biased, K, dim=-1, sorted=False)  # (M, K)
        topk_w = scores.gather(1, topk_idx)
        topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-20)
        topk_w = topk_w * self.routed_scaling_factor

        if self.training and M > 0:
            self._update_bias(scores.detach())

        topk_idx = topk_idx.view(*leading, K)
        topk_w = topk_w.view(*leading, K)
        return topk_idx, topk_w

    @torch.no_grad()
    def _update_bias(self, scores: torch.Tensor) -> None:
        """One QB step (Eq. 14): bias ← −quantile(margins) centered.

        scores: (M, E) — sigmoid router scores from this batch (detached).
        """
        M, E = scores.shape
        K = self.n_experts_per_tok
        if M < 1 or E < 1:
            return

        # Token cutoffs α_i: (K+1)-th largest of (s_i + b)  → index K in desc sort
        biased = scores + self.expert_bias
        k_take = min(K + 1, E)
        alpha = torch.topk(biased, k_take, dim=-1).values[:, -1]   # (M,)

        # Expert-side: (1 - K/E)-quantile of margins s_{:,j} - α
        # ≈ (q)-th largest with q = floor(M * K / E)
        margins = scores - alpha.unsqueeze(-1)                     # (M, E)
        q = max(int(M * K / E), 0)
        q = min(q, M - 1)
        sorted_m, _ = torch.sort(margins, dim=0, descending=True)  # (M, E)
        beta = sorted_m[q]                                         # (E,)  ≈ quantile
        b_new = -beta
        b_new = b_new - b_new.mean()
        self.expert_bias.copy_(b_new)
