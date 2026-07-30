"""
kimi_k3/residuals/depth_history.py — depth-wise memory for Block AttnRes.

Paper: "Attention Residuals" (arXiv:2603.15031). Block AttnRes keeps:
  - completed: sealed block representations (b_0 = token embedding, then b_1..)
  - partial:   running sum of sublayer outputs inside the current block

block_size counts decoder layers. A block is sealed at the beginning of layer
indices divisible by `block_size`, matching Moonshot's reference flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class DepthHistory:
    """Running Block AttnRes memory across transformer depth."""

    completed: list[Tensor] = field(default_factory=list)  # each (B, T, d)
    partial: Tensor | None = None                          # (B, T, d) or None
    sublayer_count: int = 0                                # compatibility/debug counter

    @classmethod
    def from_embedding(cls, embed: Tensor) -> DepthHistory:
        """b_0 = token embedding; first mix attends over embed alone."""
        return cls(completed=[embed], partial=None, sublayer_count=0)

    def sources(self) -> list[Tensor]:
        """Values for depth attention: completed blocks (+ partial if active)."""
        src = list(self.completed)
        if self.partial is not None:
            src.append(self.partial)
        return src

    def accumulate(self, sublayer_out: Tensor) -> None:
        """Add a sublayer output into the current block partial sum."""
        if self.partial is None:
            self.partial = sublayer_out
        else:
            self.partial = self.partial + sublayer_out
        self.sublayer_count += 1

    def start_layer(self, layer_idx: int, block_size: int) -> None:
        """Seal the preceding layer block at an official AttnRes boundary."""
        if (
            layer_idx > 0
            and layer_idx % block_size == 0
            and self.partial is not None
        ):
            self.completed.append(self.partial)
            self.partial = None

    def seal_remainder(self) -> None:
        """After the last layer, fold any open partial into completed."""
        if self.partial is not None:
            self.completed.append(self.partial)
            self.partial = None
