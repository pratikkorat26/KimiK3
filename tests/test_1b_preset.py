"""tests/test_1b_preset.py — the kimi_1b_64k training-target preset.

Contract:
  1. total params land in [0.9B, 1.1B] (the "~1B total" goal)
  2. 64K context, full hybrid schedule (KDA + Gated MLA), MoE active
  3. active-param accounting is internally consistent
"""

from kimi_k3 import KimiK3Model, ModelConfig
from kimi_k3.moe.latent_moe import StableLatentMoE


def test_1b_preset_total_params():
    cfg = ModelConfig.kimi_1b_64k()
    model = KimiK3Model(cfg)
    total = sum(p.numel() for p in model.parameters())
    assert 0.9e9 <= total <= 1.1e9, f"got {total / 1e9:.3f}B params"


def test_1b_preset_shape_and_schedule():
    cfg = ModelConfig.kimi_1b_64k()
    assert cfg.max_seq_len == 65_536
    assert cfg.use_interim_ffn is False
    assert cfg.use_interim_residual is False
    # 3 KDA : 1 MLA, final layer MLA
    assert cfg.attention_type(0) == "kda"
    assert cfg.attention_type(3) == "gated_mla"
    assert cfg.attention_type(cfg.n_layer - 1) == "gated_mla"
    assert cfg.n_experts_per_tok < cfg.n_experts


def test_1b_preset_active_less_than_total():
    cfg = ModelConfig.kimi_1b_64k()
    model = KimiK3Model(cfg)
    total = sum(p.numel() for p in model.parameters())

    inactive = 0
    for module in model.modules():
        if isinstance(module, StableLatentMoE):
            per_expert = sum(p.numel() for p in module.routed_experts[0].parameters())
            inactive += per_expert * (cfg.n_experts - cfg.n_experts_per_tok)
    active = total - inactive
    # MoE means strictly fewer active than total, but attention+embeddings keep
    # the active fraction high at 1B scale.
    assert 0 < active < total
    assert active / total > 0.3
