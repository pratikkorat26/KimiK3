"""
kimi_k3/mtp/heads.py — sequential Multi-Token Prediction (MTP) heads.

Faithful to the DeepSeek-V3 / Kimi K2/K3 MTP scheme (config alias
`num_nextn_predict_layers`, arXiv:2507.20534 lineage): D chained modules, each
teacher-forced on the ground-truth future token and refining the previous
module's hidden state, all sharing the trunk's `embed_tokens` + `lm_head`.

Deviation (documented, for CPU-testability): each module refines with a
lightweight SiTU-GLU FFN rather than a full KDA/MLA transformer block. This
keeps the extra training signal while staying transparent and cheap to test.

MTP is a training-time construct only — it produces auxiliary logits used by the
loss (see kimi_k3/training/loss.py:mtp_loss). Speculative decoding is out of scope.

Shapes:
    trunk_hidden: (B, T, d)   — normed final hidden state of the main trunk (h_0)
    tokens:       (B, T)      — input ids (used to teacher-force future tokens)
    logits_k:     (B, T, vocab) for each head k = 1..D
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig
from ..moe.situ_glu import SiTUGLU
from ..norms import rms_norm


class _MTPModule(nn.Module):
    """One MTP step: combine [RMSNorm(h_{k-1}); RMSNorm(emb_future)] → FFN refine."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.hidden_size
        self.hnorm = rms_norm(d, cfg.eps)   # on the previous hidden state
        self.enorm = rms_norm(d, cfg.eps)   # on the future-token embedding
        self.proj = nn.Linear(2 * d, d, bias=False)
        self.ffn = SiTUGLU(
            d,
            cfg.moe_intermediate_size,
            beta_gate=cfg.situ_beta_gate,
            beta_up=cfg.situ_beta_up,
        )
        self.onorm = rms_norm(d, cfg.eps)   # final norm before the shared lm_head

    def forward(self, hidden: torch.Tensor, future_embed: torch.Tensor) -> torch.Tensor:
        combined = torch.cat((self.hnorm(hidden), self.enorm(future_embed)), dim=-1)
        h = self.proj(combined)             # (B, T, d)
        h = h + self.ffn(h)                 # residual refinement
        return self.onorm(h)


class MTPHeads(nn.Module):
    """D sequential next-n-predict heads sharing the trunk embedding + lm_head."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.depth = cfg.num_nextn_predict_layers
        self.modules_ = nn.ModuleList([_MTPModule(cfg) for _ in range(self.depth)])

    def forward(
        self,
        trunk_hidden: torch.Tensor,
        tokens: torch.Tensor,
        embed_tokens: nn.Embedding,
        lm_head: nn.Linear,
    ) -> list[torch.Tensor]:
        """Return [logits_1, ..., logits_D]; head k predicts token at offset 1+k.

        Head k is teacher-forced on emb(t_{i+k}): the token embeddings shifted
        left by k so position i attends to its k-th future input. The trailing k
        positions have no valid future token — their logits are ignored by the
        loss (which shifts targets to match).
        """
        if self.depth == 0:
            return []
        emb = embed_tokens(tokens)          # (B, T, d)
        hidden = trunk_hidden               # h_0
        outputs: list[torch.Tensor] = []
        for k, module in enumerate(self.modules_, start=1):
            # emb(t_{i+k}): shift left by k, pad the tail with zeros (masked in loss).
            future_embed = torch.zeros_like(emb)
            if k < emb.shape[1]:
                future_embed[:, : emb.shape[1] - k] = emb[:, k:]
            hidden = module(hidden, future_embed)
            outputs.append(lm_head(hidden))  # (B, T, vocab)
        return outputs
