"""
kimi_k3/moe/situ_glu.py — Sigmoid Tanh Unit GLU (SiTU-GLU).

Kimi K3 replaces SwiGLU with SiTU-GLU to bound activations under extreme MoE
sparsity (arXiv:2607.24653, §2.3.2 / Appendix B):

    g, u = W_gate x, W_up x
    h = [β1 tanh(g/β1) ⊙ sigmoid(g)] ⊙ [β2 tanh(u/β2)]
    out = W_down h

Defaults: β1 = 4 (gate), β2 = 25 (up) → |h| ≤ β1 β2 = 100.
Near the origin SiTU-GLU matches SwiGLU to first order.
"""

import torch
import torch.nn as nn


def situ(x: torch.Tensor, beta: float) -> torch.Tensor:
    """Scaled tanh cap: β * tanh(x / β). Shape preserved."""
    return beta * torch.tanh(x / beta)


class SiTU(nn.Module):
    """Sigmoid Tanh Unit on a single branch: situ(x, beta) * sigmoid(x)."""

    def __init__(self, beta: float = 4.0):
        super().__init__()
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return situ(x, self.beta) * torch.sigmoid(x)


class SiTUGLU(nn.Module):
    """Gated FFN expert body with SiTU-GLU activation.

    Shapes:
        x:   (..., d_in)
        out: (..., d_in)   — down-projects back to d_in (hidden or latent width)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        beta_gate: float = 4.0,
        beta_up: float = 25.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.beta_gate = beta_gate
        self.beta_up = beta_up
        self.w_gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w_up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w_down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.w_gate(x)                                         # (..., intermediate)
        u = self.w_up(x)                                           # (..., intermediate)
        dtype = g.dtype
        g, u = g.float(), u.float()
        # gate branch: β1 tanh(g/β1) ⊙ σ(g);  up branch: β2 tanh(u/β2)
        h = (situ(g, self.beta_gate) * torch.sigmoid(g)) * situ(u, self.beta_up)
        return self.w_down(h.to(dtype))                            # (..., hidden)
