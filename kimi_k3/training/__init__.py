"""
kimi_k3/training/ — minimal training loop for KimiK3Model.

Phase 2+: causal-LM cross-entropy loss (+ MTP auxiliary loss), a TrainConfig,
and a Trainer (warmup+cosine LR, grad clipping, checkpointing, overfit helper)
supporting either AdamW or the Per-Head Muon hybrid optimizer. CPU by default,
MPS/CUDA opt-in. The progressive-length curriculum remains future work
(see docs/roadmap.md).
"""

from .config import TrainConfig
from .loss import causal_lm_loss, mtp_loss, perplexity
from .muon import Muon, build_muon_optimizer, zeropower_via_newtonschulz5
from .trainer import Trainer, resolve_device

__all__ = [
    "Muon",
    "TrainConfig",
    "Trainer",
    "build_muon_optimizer",
    "causal_lm_loss",
    "mtp_loss",
    "perplexity",
    "resolve_device",
    "zeropower_via_newtonschulz5",
]
