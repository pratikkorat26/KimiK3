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


def mtp_loss(
    mtp_logits: list[Tensor],
    labels: Tensor,
    weight: float,
    ignore_index: int = -100,
) -> Tensor:
    """Weighted auxiliary loss over the Multi-Token Prediction heads.

    `labels` follow the same already-shifted convention as `causal_lm_loss`:
    labels[t] is the offset-1 target for position t. MTP head j (j = 1..D)
    predicts the token at offset 1+j, so its target is labels shifted a further
    j positions: CE(mtp_logits[j-1][:, :T-j], labels[:, j:]). The last j
    positions of each head have no valid target and are dropped by the shift.

    Returns weight / D * sum_j CE_j (a zero scalar when there are no heads).
    """
    if not mtp_logits or weight == 0.0:
        example = mtp_logits[0] if mtp_logits else labels
        return torch.zeros((), device=example.device, dtype=torch.float32)
    depth = len(mtp_logits)
    vocab = mtp_logits[0].shape[-1]
    total = mtp_logits[0].new_zeros(())
    for j, logits_j in enumerate(mtp_logits, start=1):
        if labels.shape[1] <= j:
            continue
        total = total + F.cross_entropy(
            logits_j[:, : labels.shape[1] - j].reshape(-1, vocab),
            labels[:, j:].reshape(-1),
            ignore_index=ignore_index,
        )
    return (weight / depth) * total


@torch.no_grad()
def perplexity(loss: Tensor) -> float:
    """exp(loss), clamped to avoid overflow on an untrained model."""
    return float(torch.exp(loss.detach().clamp(max=20.0)))
