"""
kimi_k3/moe/latent_moe.py — Stable LatentMoE + interim DenseFFN.

Kimi K3 channel-mixing (arXiv:2607.24653):
  - shared SiTU-GLU experts at full hidden width
  - routed path: w_down → latent → top-k SiTU-GLU experts → RMSNorm → w_up
  - Quantile Balancing router (no aux load-balancing loss)

DenseFFN remains only for tiny() / tiny_hybrid() CPU unit tests.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .router import QuantileBalancingRouter
from .situ_glu import SiTUGLU


class DenseFFN(nn.Module):
    """INTERIM dense SwiGLU — tiny-preset placeholder only."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class StableLatentMoE(nn.Module):
    """Stable LatentMoE: shared full-width experts + routed latent experts.

    Shapes:
        x:   (B, T, d)
        out: (B, T, d)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_size
        d_lat = cfg.latent_size
        d_ff = cfg.moe_intermediate_size
        beta_g, beta_u = cfg.situ_beta_gate, cfg.situ_beta_up

        self.w_down = nn.Linear(d, d_lat, bias=False)
        self.w_up = nn.Linear(d_lat, d, bias=False)
        self.routed_norm = nn.RMSNorm(d_lat, eps=cfg.eps)
        self.router = QuantileBalancingRouter(cfg)

        self.routed_experts = nn.ModuleList(
            [
                SiTUGLU(d_lat, d_ff, beta_gate=beta_g, beta_up=beta_u)
                for _ in range(cfg.n_experts)
            ]
        )
        self.shared_experts = SiTUGLU(
            d,
            d_ff * cfg.n_shared_experts,
            beta_gate=beta_g,
            beta_up=beta_u,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        # --- shared experts (full width) ------------------------------------
        shared = self.shared_experts(x)
        # shared: (B, T, d)

        # --- routed latent path ---------------------------------------------
        latent = self.w_down(x)                                    # (B, T, d_lat)
        topk_idx, topk_w = self.router(x)                          # (B, T, K)
        routed = self._dispatch_routed(latent, topk_idx, topk_w)   # (B, T, d_lat)
        routed = self.w_up(self.routed_norm(routed))               # (B, T, d)

        return shared + routed

    def _dispatch_routed(
        self,
        latent: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_w: torch.Tensor,
    ) -> torch.Tensor:
        """Gather–expert–scatter for top-k assignments (study-scale loop).

        latent:   (B, T, d_lat)
        topk_idx: (B, T, K)
        topk_w:   (B, T, K)
        return:   (B, T, d_lat)
        """
        B, T, d_lat = latent.shape
        K = topk_idx.shape[-1]
        flat = latent.reshape(B * T, d_lat)                        # (M, d_lat)
        idx = topk_idx.reshape(B * T, K)                           # (M, K)
        w = topk_w.reshape(B * T, K)                               # (M, K)
        M = flat.shape[0]
        out = flat.new_zeros(M, d_lat)

        for expert_id, expert in enumerate(self.routed_experts):
            # tokens that selected this expert in any of their K slots
            mask = idx == expert_id                                # (M, K)
            if not mask.any():
                continue
            token_hit = mask.any(dim=-1)                           # (M,)
            token_ids = token_hit.nonzero(as_tuple=False).squeeze(-1)
            # weight = sum of topk weights for this expert across K slots
            weights = (w * mask.float()).sum(dim=-1)[token_ids]    # (n_tok,)
            y = expert(flat[token_ids])                            # (n_tok, d_lat)
            out.index_add_(0, token_ids, y * weights.unsqueeze(-1))

        return out.view(B, T, d_lat)


def build_ffn(cfg: ModelConfig, layer_idx: int) -> nn.Module:
    """Dense first layers / interim DenseFFN, else StableLatentMoE."""
    if cfg.use_interim_ffn:
        return DenseFFN(cfg.hidden_size, cfg.intermediate_size)
    if layer_idx < cfg.n_dense_layers:
        return SiTUGLU(
            cfg.hidden_size,
            cfg.intermediate_size,
            beta_gate=cfg.situ_beta_gate,
            beta_up=cfg.situ_beta_up,
        )
    return StableLatentMoE(cfg)
