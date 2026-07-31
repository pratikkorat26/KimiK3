"""
kimi_k3 — readable, study-scale PyTorch reference for the Kimi K3 architecture.

Implemented: Kimi Delta Attention (KDA) and Gated MLA (NoPE + query LoRA),
Stable LatentMoE, SiTU-GLU, Block AttnRes, and class-based decode caches.
Deferred: MoonViT-V2 vision.

The modules favor transparent math and validation over production kernels.

    from kimi_k3 import KimiK3Model, ModelConfig, AttentionCache

    model = KimiK3Model(ModelConfig.tiny_hybrid())
    logits, cache = model(tokens, mode="chunk", use_cache=True)
"""

from .attention import (
    AttentionCache,
    GatedMLA,
    KDACache,
    KimiDeltaAttention,
    MLACache,
    kda_chunkwise,
    kda_recurrence,
)
from .config import KDAConfig, ModelConfig
from .model import KimiK3Model
from .moe import DenseFFN, QuantileBalancingRouter, SiTUGLU, StableLatentMoE
from .residuals import BlockAttnRes, DepthHistory

__all__ = [
    "AttentionCache",
    "BlockAttnRes",
    "DenseFFN",
    "DepthHistory",
    "GatedMLA",
    "KDACache",
    "KDAConfig",
    "KimiDeltaAttention",
    "KimiK3Model",
    "MLACache",
    "ModelConfig",
    "QuantileBalancingRouter",
    "SiTUGLU",
    "StableLatentMoE",
    "kda_chunkwise",
    "kda_recurrence",
]

# Hugging Face is an optional pretraining dependency. Keep the core package
# importable for lightweight architecture tests without transformers installed.
try:
    from .hf import KimiK3ForCausalLM, KimiK3HFConfig

    __all__ += ["KimiK3ForCausalLM", "KimiK3HFConfig"]
except ImportError:
    pass
