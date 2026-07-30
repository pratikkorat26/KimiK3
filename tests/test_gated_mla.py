"""
tests/test_gated_mla.py — numerical verification of GatedMLA + hybrid model.

Contract:
  1. output / MLACache shapes
  2. cached prefill + decode ≈ full forward
  3. backward
  4. tiny_hybrid: 3 KDA + 1 MLA with AttentionCache
"""

import pytest
import torch

from kimi_k3 import (
    AttentionCache,
    GatedMLA,
    KDACache,
    KimiK3Model,
    MLACache,
    ModelConfig,
)

TOL = 1e-5


def make_layer():
    torch.manual_seed(0)
    cfg = ModelConfig.tiny_hybrid()
    return cfg, GatedMLA(cfg).eval()


def test_output_shapes():
    cfg, layer = make_layer()
    x = torch.randn(2, 17, cfg.hidden_size)
    out, cache = layer(x, use_cache=True)

    assert out.shape == (2, 17, cfg.hidden_size)
    assert isinstance(cache, MLACache)
    assert cache.kv_latent.shape == (2, 17, cfg.kv_latent_dim)
    assert cache.k_shared.shape == (2, 17, cfg.qk_shared_dim)
    assert cache.seq_len == 17


def test_cache_continuation():
    cfg, layer = make_layer()
    x = torch.randn(1, 17, cfg.hidden_size)
    with torch.no_grad():
        ref, _ = layer(x, use_cache=False)
        pre, cache = layer(x[:, :10], use_cache=True)
        outs = [pre]
        for t in range(10, 17):
            o, cache = layer(x[:, t : t + 1], cache=cache, use_cache=True)
            outs.append(o)
        cont = torch.cat(outs, dim=1)

    assert (ref - cont).abs().max().item() < TOL
    assert cache.seq_len == 17


@pytest.mark.parametrize("appended_length", [2, 7])
def test_cache_multi_token_continuation_is_causal(appended_length):
    cfg, layer = make_layer()
    prefix_length = 10
    x = torch.randn(1, prefix_length + appended_length, cfg.hidden_size)
    with torch.no_grad():
        ref, _ = layer(x, use_cache=False)
        _, cache = layer(x[:, :prefix_length], use_cache=True)
        tail, cache = layer(
            x[:, prefix_length:],
            cache=cache,
            use_cache=True,
        )

    assert torch.allclose(
        ref[:, prefix_length:],
        tail,
        atol=TOL,
        rtol=TOL,
    )
    assert cache.seq_len == prefix_length + appended_length


def test_backward_runs():
    cfg, layer = make_layer()
    layer.train()
    x = torch.randn(1, 17, cfg.hidden_size, requires_grad=True)
    out, _ = layer(x)
    out.sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_layer_rejects_malformed_cache():
    cfg, layer = make_layer()
    x = torch.randn(2, 1, cfg.hidden_size)
    cache = MLACache(
        kv_latent=torch.zeros(1, 3, cfg.kv_latent_dim),
        k_shared=torch.zeros(1, 3, cfg.qk_shared_dim),
    )
    with pytest.raises(ValueError, match="batch size"):
        layer(x, cache=cache)


def test_tiny_hybrid_model_forward():
    torch.manual_seed(0)
    cfg = ModelConfig.tiny_hybrid()
    types = [cfg.attention_type(i) for i in range(cfg.n_layer)]
    assert types == ["kda", "kda", "kda", "gated_mla"]

    model = KimiK3Model(cfg).eval()
    assert isinstance(model.blocks[3].attn, GatedMLA)
    tokens = torch.randint(0, cfg.vocab_size, (1, 8))
    logits, cache = model(tokens, mode="chunk", use_cache=True)

    assert logits.shape == (1, 8, cfg.vocab_size)
    assert isinstance(cache, AttentionCache)
    assert len(cache) == cfg.n_layer
    assert isinstance(cache.get(0), KDACache)
    assert isinstance(cache.get(3), MLACache)
    assert cache.get(3).kv_latent.shape == (1, 8, cfg.kv_latent_dim)
    assert cache.get(3).k_shared.shape == (1, 8, cfg.qk_shared_dim)


@pytest.mark.parametrize("appended_length", [1, 4])
@pytest.mark.parametrize("mode", ["chunk", "recurrent"])
def test_tiny_hybrid_model_cached_continuation(appended_length, mode):
    torch.manual_seed(0)
    cfg = ModelConfig.tiny_hybrid()
    model = KimiK3Model(cfg).eval()
    prefix_length = 6
    tokens = torch.randint(
        0,
        cfg.vocab_size,
        (1, prefix_length + appended_length),
    )

    with torch.no_grad():
        ref, _ = model(tokens, mode="chunk")
        _, cache = model(
            tokens[:, :prefix_length],
            mode="chunk",
            use_cache=True,
        )
        tail, cache = model(
            tokens[:, prefix_length:],
            mode=mode,
            cache=cache,
            use_cache=True,
        )

    assert torch.allclose(
        ref[:, prefix_length:],
        tail,
        atol=2e-5,
        rtol=2e-5,
    )
    assert cache.tokens_seen == prefix_length + appended_length
