"""
kimi_k3/mtp/ — Multi-Token Prediction heads.

Kimi K2/K3 attach extra "next-n-predict" layers (config alias
`num_nextn_predict_layers`) that predict several future tokens per step —
useful as extra training signal (and, elsewhere, for speculative decoding).

Implemented here as sequential, DeepSeek/K3-faithful heads (`MTPHeads`): D
chained modules teacher-forced on ground-truth future tokens, sharing the
trunk's embedding + lm_head, each refined by a lightweight SiTU-GLU FFN. Enable
via `ModelConfig.num_nextn_predict_layers` (> 0) and weight the auxiliary loss
with `ModelConfig.mtp_loss_weight`. Training-only; speculative decoding is out
of scope.
"""

from .heads import MTPHeads

__all__ = ["MTPHeads"]
