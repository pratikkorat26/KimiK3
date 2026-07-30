"""Checks for the released Kimi K3 text-architecture contract."""

from dataclasses import replace

import pytest
import torch

from kimi_k3 import GatedMLA, KimiDeltaAttention, KimiK3Model, ModelConfig
from kimi_k3.moe import QuantileBalancingRouter


def test_full_config_matches_released_text_model():
    cfg = ModelConfig()

    assert cfg.vocab_size == 163_840
    assert cfg.hidden_size == 7168
    assert cfg.n_layer == 93
    assert cfg.n_heads == 96
    assert cfg.intermediate_size == 33_792
    assert cfg.q_lora_rank == 1536
    assert cfg.kv_latent_dim == 512
    assert cfg.eps == 1e-5
    assert cfg.initializer_range == 0.02
    assert cfg.kda_gate_lower_bound == -5.0
    assert cfg.kda_use_full_rank_gate
    assert cfg.routed_scaling_factor == 1.0

    schedule = [cfg.attention_type(i) for i in range(cfg.n_layer)]
    assert schedule.count("kda") == 69
    assert schedule.count("gated_mla") == 24
    assert schedule[-1] == "gated_mla"


def test_mla_uses_query_lora_and_full_rank_output_gate():
    cfg = ModelConfig.tiny_hybrid()
    layer = GatedMLA(cfg)

    assert layer.w_q_down.weight.shape == (cfg.q_lora_rank, cfg.hidden_size)
    assert layer.w_q_up.weight.shape == (
        cfg.n_heads * cfg.q_head_dim,
        cfg.q_lora_rank,
    )
    assert layer.w_gate.weight.shape == (
        cfg.n_heads * cfg.v_head_dim,
        cfg.hidden_size,
    )


def test_kda_uses_reference_gate_parameterization():
    cfg = ModelConfig.tiny().kda_config()
    layer = KimiDeltaAttention(cfg)

    assert layer.A_log.shape == (cfg.num_heads,)
    assert layer.dt_bias.shape == (cfg.num_heads * cfg.head_dim_k,)
    assert torch.all((layer.A_log.exp() >= 1) & (layer.A_log.exp() <= 16))

    x = torch.randn(2, 5, cfg.hidden_size)
    gate = layer._decay_gate(x)
    assert gate.dtype == torch.float32
    assert torch.all(gate >= cfg.gate_lower_bound)
    assert torch.all(gate < 0)


def test_router_selects_with_bias_but_weights_unbiased_sigmoid_scores():
    cfg = replace(
        ModelConfig.tiny(),
        hidden_size=4,
        latent_size=2,
        n_experts=3,
        n_experts_per_tok=2,
    )
    router = QuantileBalancingRouter(cfg).eval()
    with torch.no_grad():
        router.w_router.weight.zero_()
        router.expert_bias.copy_(torch.tensor([0.0, 1.0, 2.0]))

    idx, weights = router(torch.ones(1, 1, cfg.hidden_size))

    assert set(idx.flatten().tolist()) == {1, 2}
    assert torch.allclose(weights, torch.full_like(weights, 0.5))


def test_model_initialization_matches_reference_std():
    torch.manual_seed(0)
    cfg = ModelConfig.tiny_hybrid()
    model = KimiK3Model(cfg)

    assert model.embed_tokens.weight.std().item() == pytest.approx(
        cfg.initializer_range, rel=0.05
    )
