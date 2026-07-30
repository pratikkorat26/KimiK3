"""
kimi_k3/training/loss.py — causal language-modeling loss.

Plain next-token cross-entropy over the model's (B, T, vocab) logits. No model
change is needed: the trainer calls model(tokens, mode="chunk") and feeds the
returned logits here. `ignore_index` masks padding / positions to skip.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def causal_lm_loss(logits: Tensor, labels: Tensor, ignore_index: int = -100) -> Tensor:
    """Mean cross-entropy of next-token predictions.

    logits: (B, T, vocab); labels: (B, T) int64. Labels are already shifted
    (label[t] is the target for position t), so no shifting happens here.
    """
    vocab = logits.shape[-1]
    return F.cross_entropy(
        logits.reshape(-1, vocab),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )


@torch.no_grad()
def perplexity(loss: Tensor) -> float:
    """exp(loss), clamped to avoid overflow on an untrained model."""
    return float(torch.exp(loss.detach().clamp(max=20.0)))
