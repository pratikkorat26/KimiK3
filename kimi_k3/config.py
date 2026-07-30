"""
kimi_k3/config.py — KDA layer config and the full Kimi K3 ModelConfig.

Official K3 numbers from MoonshotAI/Kimi-K3 model card:
  93 layers · hidden 7168 · 96 heads · vocab 163840 · context 1M
  69 KDA + 24 Gated MLA · LatentMoE 3584 / 896 experts / 16 active + 2 shared
  SiTU-GLU · Block AttnRes · MoonViT-V2 (vision deferred)

Use ModelConfig() for full-scale dims, ModelConfig.tiny() / tiny_hybrid() for
fast CPU unit tests, and ModelConfig.small() for a ~100M K3-like runnable model.

MLA config names (ours) vs Moonshot aliases (comments only):
  kv_latent_dim   ↔ kv_lora_rank
  qk_content_dim  ↔ qk_nope_head_dim
  qk_shared_dim   ↔ qk_rope_head_dim  (NoPE: not RoPE)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass
class KDAConfig:
    """Hyperparameters of a single KDA layer.

    Notation (used in every shape comment across this package):
        B   batch size
        T   sequence length (number of tokens)
        d   model hidden size (layer input/output width)
        H   number of attention heads
        d_k key/query head dimension
        d_v value head dimension
        C   chunk size        (only used by the chunkwise algorithm)
        N   number of chunks  (N = T / C after padding, chunkwise only)
    """

    hidden_size: int = 1024      # d  — layer input/output width: x, o are (B, T, d)
    num_heads: int = 8           # H  — KDA heads; per-head tensors are (B, H, T, ·)
    head_dim_k: int = 128        # d_k — q/k head dim; paper Section 3.3 sets 128
    head_dim_v: int = 128        # d_v — v head dim; paper Section 3.3 sets 128
    conv_kernel_size: int = 4    # K  — ShortConv window on q/k/v (follows GDN, ref [111])
    chunk_size: int = 64         # C  — chunk length of the chunkwise algorithm (paper: 64)
    gate_rank: int = 128         # r  — low-rank of W_down/W_up for the decay gate
    gate_lower_bound: float = -5.0
    use_full_rank_gate: bool = True
    eps: float = 1e-5            # epsilon for the head-wise RMSNorm and L2Norm
    use_q_scale: bool = True     # scale q by d_k**-0.5, as in the kernel pseudocode

    def __post_init__(self) -> None:
        positive_ints = (
            "hidden_size",
            "num_heads",
            "head_dim_k",
            "head_dim_v",
            "conv_kernel_size",
            "chunk_size",
            "gate_rank",
        )
        for name in positive_ints:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if not isinstance(self.eps, (int, float)) or isinstance(self.eps, bool) or self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps!r}")
        if (
            not isinstance(self.gate_lower_bound, (int, float))
            or isinstance(self.gate_lower_bound, bool)
            or not -5.0 <= self.gate_lower_bound < 0.0
        ):
            raise ValueError(
                "gate_lower_bound must be in the safe range [-5, 0), "
                f"got {self.gate_lower_bound!r}"
            )


AttentionType = Literal["kda", "gated_mla"]


@dataclass
class ModelConfig:
    """Configuration of the assembled KimiK3Model.

    Defaults match the official Kimi K3 model card. For a CPU-studyable
    all-KDA backbone, use ModelConfig.tiny(); for 3:1 hybrid, tiny_hybrid().
    """

    # --- core transformer ---
    vocab_size: int = 163_840
    hidden_size: int = 7168          # d — attention hidden dimension
    n_layer: int = 93
    n_heads: int = 96                # H
    head_dim_k: int = 128            # d_k (Kimi Linear / KDA convention)
    head_dim_v: int = 128            # d_v
    max_seq_len: int = 1_048_576     # 1M context
    eps: float = 1e-5
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False

    # --- hybrid attention (69 KDA + 24 Gated MLA) ---
    force_all_kda: bool = False
    conv_kernel_size: int = 4
    chunk_size: int = 64
    gate_rank: int = 128             # low-rank width for KDA decay projection
    kda_gate_lower_bound: float = -5.0
    kda_use_full_rank_gate: bool = True

    # --- Gated MLA (Kimi Linear NoPE structure; clear names, not LoRA/RoPE jargon) ---
    kv_latent_dim: int = 512         # width of c_kv  (Moonshot: kv_lora_rank)
    q_lora_rank: int = 1536
    qk_content_dim: int = 128        # per-head Q/K content (Moonshot: qk_nope_head_dim)
    qk_shared_dim: int = 64          # per-head shared Q/K (Moonshot: qk_rope_head_dim; NoPE)
    v_head_dim: int = 128            # per-head value dim

    # --- Stable LatentMoE ---
    latent_size: int = 3584
    moe_intermediate_size: int = 3072
    n_experts: int = 896
    n_experts_per_tok: int = 16
    n_shared_experts: int = 2
    routed_scaling_factor: float = 1.0
    n_dense_layers: int = 1
    intermediate_size: int = 33792   # dense first-layer FFN width
    situ_beta_gate: float = 4.0      # β1 in SiTU-GLU
    situ_beta_up: float = 25.0       # β2 in SiTU-GLU

    # --- Block AttnRes ---
    attn_res_block_size: int = 12

    # --- study / interim switches ---
    use_interim_ffn: bool = False
    use_interim_residual: bool = False

    def __post_init__(self) -> None:
        positive_ints = (
            "vocab_size",
            "hidden_size",
            "n_layer",
            "n_heads",
            "head_dim_k",
            "head_dim_v",
            "max_seq_len",
            "conv_kernel_size",
            "chunk_size",
            "gate_rank",
            "kv_latent_dim",
            "q_lora_rank",
            "qk_content_dim",
            "qk_shared_dim",
            "v_head_dim",
            "latent_size",
            "moe_intermediate_size",
            "n_experts",
            "n_experts_per_tok",
            "n_shared_experts",
            "intermediate_size",
            "attn_res_block_size",
        )
        for name in positive_ints:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

        if (
            not isinstance(self.n_dense_layers, int)
            or isinstance(self.n_dense_layers, bool)
            or self.n_dense_layers < 0
        ):
            raise ValueError(
                f"n_dense_layers must be a non-negative integer, got {self.n_dense_layers!r}"
            )
        if self.n_dense_layers > self.n_layer:
            raise ValueError(
                "n_dense_layers cannot exceed n_layer "
                f"({self.n_dense_layers} > {self.n_layer})"
            )
        if self.n_experts_per_tok > self.n_experts:
            raise ValueError(
                "n_experts_per_tok cannot exceed n_experts "
                f"({self.n_experts_per_tok} > {self.n_experts})"
            )

        positive_floats = (
            "eps",
            "initializer_range",
            "routed_scaling_factor",
            "situ_beta_gate",
            "situ_beta_up",
        )
        for name in positive_floats:
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be positive, got {value!r}")
        if (
            not isinstance(self.kda_gate_lower_bound, (int, float))
            or isinstance(self.kda_gate_lower_bound, bool)
            or not -5.0 <= self.kda_gate_lower_bound < 0.0
        ):
            raise ValueError(
                "kda_gate_lower_bound must be in the safe range [-5, 0), "
                f"got {self.kda_gate_lower_bound!r}"
            )

    @property
    def q_head_dim(self) -> int:
        """Per-head query/key width = content + shared."""
        return self.qk_content_dim + self.qk_shared_dim

    def attention_type(self, layer_idx: int) -> AttentionType:
        """Hybrid schedule: 3 KDA : 1 Gated MLA; final layer always MLA."""
        if self.force_all_kda:
            return "kda"
        if layer_idx == self.n_layer - 1 or (layer_idx + 1) % 4 == 0:
            return "gated_mla"
        return "kda"

    def kda_config(self) -> KDAConfig:
        return KDAConfig(
            hidden_size=self.hidden_size,
            num_heads=self.n_heads,
            head_dim_k=self.head_dim_k,
            head_dim_v=self.head_dim_v,
            conv_kernel_size=self.conv_kernel_size,
            chunk_size=self.chunk_size,
            gate_rank=self.gate_rank,
            gate_lower_bound=self.kda_gate_lower_bound,
            use_full_rank_gate=self.kda_use_full_rank_gate,
            eps=self.eps,
        )

    @classmethod
    def tiny(cls) -> "ModelConfig":
        """Small all-KDA study preset — interim FFN / residual."""
        return cls(
            vocab_size=4096,
            hidden_size=256,
            n_layer=4,
            n_heads=2,
            head_dim_k=64,
            head_dim_v=64,
            max_seq_len=512,
            force_all_kda=True,
            chunk_size=16,
            gate_rank=64,
            kv_latent_dim=64,
            q_lora_rank=64,
            qk_content_dim=32,
            qk_shared_dim=16,
            v_head_dim=32,
            latent_size=128,
            moe_intermediate_size=512,
            n_experts=8,
            n_experts_per_tok=2,
            n_shared_experts=1,
            n_dense_layers=0,
            intermediate_size=512,
            attn_res_block_size=4,
            use_interim_ffn=True,
            use_interim_residual=False,  # real Block AttnRes
        )

    @classmethod
    def tiny_hybrid(cls) -> "ModelConfig":
        """Small 3:1 KDA:Gated-MLA study preset (4 layers → 3 KDA + 1 MLA)."""
        return cls(
            vocab_size=4096,
            hidden_size=256,
            n_layer=4,
            n_heads=2,
            head_dim_k=64,
            head_dim_v=64,
            max_seq_len=512,
            force_all_kda=False,
            chunk_size=16,
            gate_rank=64,
            kv_latent_dim=64,
            q_lora_rank=64,
            qk_content_dim=32,
            qk_shared_dim=16,
            v_head_dim=32,
            latent_size=128,
            moe_intermediate_size=512,
            n_experts=8,
            n_experts_per_tok=2,
            n_shared_experts=1,
            n_dense_layers=0,
            intermediate_size=512,
            attn_res_block_size=4,
            use_interim_ffn=True,
            use_interim_residual=False,  # real Block AttnRes
        )

    @classmethod
    def small(cls) -> "ModelConfig":
        """~100M-parameter K3-like preset: hybrid attn + AttnRes + LatentMoE."""
        return cls(
            vocab_size=32_000,
            hidden_size=512,
            n_layer=12,
            n_heads=8,
            head_dim_k=64,
            head_dim_v=64,
            max_seq_len=4096,
            force_all_kda=False,
            chunk_size=64,
            gate_rank=64,
            kv_latent_dim=128,
            q_lora_rank=128,
            qk_content_dim=32,
            qk_shared_dim=16,
            v_head_dim=32,
            latent_size=256,
            moe_intermediate_size=512,
            n_experts=12,  # nudged from 16 so total params land ~100M (±20%)
            n_experts_per_tok=2,
            n_shared_experts=1,
            n_dense_layers=1,
            intermediate_size=512,
            situ_beta_gate=4.0,
            situ_beta_up=25.0,
            attn_res_block_size=4,
            use_interim_ffn=False,
            use_interim_residual=False,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
