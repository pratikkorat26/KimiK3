"""
kimi_k3/training/ — minimal training loop for KimiK3Model.

Phase 2: causal-LM cross-entropy loss, a TrainConfig, and an AdamW Trainer
(warmup+cosine LR, grad clipping, checkpointing, overfit helper). CPU by default,
MPS/CUDA opt-in. Per-Head Muon, MTP heads, and the progressive-length curriculum
are future work (see docs/roadmap.md).
"""

from .config import TrainConfig
from .loss import causal_lm_loss, perplexity
from .trainer import Trainer, resolve_device

__all__ = [
    "TrainConfig",
    "Trainer",
    "causal_lm_loss",
    "perplexity",
    "resolve_device",
]
