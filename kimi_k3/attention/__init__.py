"""
Attention: KDA + Gated MLA (3:1 hybrid) with class-based caches.

Cache contract:
  KDA  → KDACache (fixed-size recurrent + ShortConv windows)
  MLA  → MLACache (growing kv_latent + k_shared)
  model → AttentionCache (one slot per layer)

mode: "chunk" | "recurrent" — used by KDA; MLA accepts and ignores for API parity.
"""

from .cache import AttentionCache, KDACache, MLACache
from .gated_mla import GatedMLA
from .kda import KimiDeltaAttention
from .kda_ops import kda_chunkwise, kda_recurrence

__all__ = [
    "AttentionCache",
    "GatedMLA",
    "KDACache",
    "KimiDeltaAttention",
    "MLACache",
    "kda_chunkwise",
    "kda_recurrence",
]
