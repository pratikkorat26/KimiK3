"""
tests/test_moe.py — SiTU-GLU, Quantile Balancing router, Stable LatentMoE, small().

Contract:
  1. SiTU-GLU shapes + bounded activation
  2. router returns top-k indices/weights; bias updates in train mode
  3. LatentMoE forward + backward
  4. ModelConfig.small() ~100M params (±20%) and end-to-end forward
"""

import torch

from kimi_k3 import (
    KimiK3Model,
    ModelConfig,
    QuantileBalancingRouter,
    SiTUGLU,
    StableLatentMoE,
)
from kimi_k3.moe.situ_glu import situ


def test_situ_glu_shapes_and_bound():
    layer = SiTUGLU(64, 128, beta_gate=4.0, beta_up=25.0)
    x = torch.randn(2, 5, 64) * 10
    y = layer(x)
    assert y.shape == (2, 5, 64)
    assert torch.isfinite(y).all()
    # scalar situ branches are bounded; product through down-proj need not be ≤100,
    # but intermediate h before down is: check situ helper
    z = torch.linspace(-50, 50, 200)
    assert situ(z, 4.0).abs().max() <= 4.0 + 1e-5
    assert situ(z, 25.0).abs().max() <= 25.0 + 1e-5


def test_router_topk_and_bias_update():
    cfg = ModelConfig.small()
    router = QuantileBalancingRouter(cfg)
    x = torch.randn(2, 7, cfg.hidden_size)
    router.train()
    bias_before = router.expert_bias.clone()
    idx, w = router(x)
    assert idx.shape == (2, 7, cfg.n_experts_per_tok)
    assert w.shape == (2, 7, cfg.n_experts_per_tok)
    assert torch.allclose(w.sum(-1), torch.ones(2, 7), atol=1e-5)
    assert not torch.equal(router.expert_bias, bias_before)


def test_latent_moe_forward_backward():
    torch.manual_seed(0)
    cfg = ModelConfig.small()
    moe = StableLatentMoE(cfg)
    x = torch.randn(2, 5, cfg.hidden_size, requires_grad=True)
    y = moe(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_small_param_count_and_forward():
    torch.manual_seed(0)
    cfg = ModelConfig.small()
    assert cfg.use_interim_ffn is False
    assert cfg.attention_type(0) == "kda"
    assert cfg.attention_type(3) == "gated_mla"

    model = KimiK3Model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    # target ~100M ±20%
    assert 80e6 <= n_params <= 120e6, f"got {n_params/1e6:.1f}M params"

    tokens = torch.randint(0, cfg.vocab_size, (1, 8))
    logits, cache = model(tokens, mode="chunk", use_cache=True)
    assert logits.shape == (1, 8, cfg.vocab_size)
    assert cache is not None
    assert torch.isfinite(logits).all()
    print(f"ModelConfig.small(): {n_params/1e6:.1f}M params")
