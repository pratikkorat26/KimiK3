"""
kimi_k3/norms.py — shared normalization helpers.

Kimi K3 uses RMSNorm at block boundaries (pre-norm) and inside KDA's
output path (head-wise). Prefer torch.nn.RMSNorm at call sites; this
module is the shared place for any project-specific norm wrappers.
"""

import torch.nn as nn


def rms_norm(dim: int, eps: float = 1e-5) -> nn.RMSNorm:
    """Factory for the standard pre-norm used in KimiK3Block / KimiK3Model."""
    return nn.RMSNorm(dim, eps=eps)
