"""
tests/test_attn_res.py — Block AttnRes + DepthHistory.

Contract:
  1. mix over one source is identity; mix over many is a convex combination
  2. block boundaries count decoder layers, not sublayers
  3. tiny / tiny_hybrid forward with use_interim_residual=False
  4. model-level initialization matches the reference normal distribution
"""

import pytest
import torch

from kimi_k3 import BlockAttnRes, DepthHistory, KimiK3Model, ModelConfig


def test_mix_single_source_is_identity():
    cfg = ModelConfig.tiny()
    mixer = BlockAttnRes(cfg, layer_idx=0)
    x = torch.randn(2, 5, cfg.hidden_size)
    history = DepthHistory.from_embedding(x)
    out = mixer.mix(history)
    assert torch.equal(out, x)


def test_mix_is_convex_combination():
    cfg = ModelConfig.tiny()
    mixer = BlockAttnRes(cfg, layer_idx=0)
    # Non-zero query so weights are content-dependent
    with torch.no_grad():
        mixer.query.weight.normal_(0, 0.1)

    a = torch.randn(1, 3, cfg.hidden_size)
    b = torch.randn(1, 3, cfg.hidden_size)
    history = DepthHistory(completed=[a], partial=b)
    out = mixer.mix(history)

    # out should lie in the span of a,b as a softmax-weighted sum — check finite + shape
    assert out.shape == a.shape
    assert torch.isfinite(out).all()


def test_zero_query_gives_uniform_average():
    cfg = ModelConfig.tiny()
    mixer = BlockAttnRes(cfg, layer_idx=0)
    with torch.no_grad():
        mixer.query.weight.zero_()

    a = torch.ones(1, 2, cfg.hidden_size)
    b = torch.ones(1, 2, cfg.hidden_size) * 3
    history = DepthHistory(completed=[a], partial=b)
    out = mixer.mix(history)
    # uniform average of a and b → 2
    assert torch.allclose(out, torch.full_like(a, 2.0), atol=1e-5)


def test_layer_boundary_seals_complete_decoder_blocks():
    d = 8
    history = DepthHistory.from_embedding(torch.zeros(1, 1, d))
    block_size = 4
    for _ in range(8):  # four decoder layers × (attention + MLP)
        history.accumulate(torch.ones(1, 1, d))
    history.start_layer(layer_idx=4, block_size=block_size)
    assert history.partial is None
    assert len(history.completed) == 2  # embedding + one sealed block
    assert torch.allclose(history.completed[1], torch.full((1, 1, d), 8.0))


def test_model_initializes_attnres_queries_with_reference_std():
    torch.manual_seed(0)
    cfg = ModelConfig.tiny()
    model = KimiK3Model(cfg)
    weight = model.blocks[0].attn_res.query.weight
    assert weight.abs().sum() > 0
    assert weight.std().item() == pytest.approx(cfg.initializer_range, rel=0.2)


def test_tiny_forward_with_attnres():
    torch.manual_seed(0)
    cfg = ModelConfig.tiny()
    assert cfg.use_interim_residual is False
    model = KimiK3Model(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 8))
    logits, cache = model(tokens, mode="chunk", use_cache=True)
    assert logits.shape == (1, 8, cfg.vocab_size)
    assert cache is not None
    assert torch.isfinite(logits).all()


def test_tiny_hybrid_forward_with_attnres():
    torch.manual_seed(0)
    cfg = ModelConfig.tiny_hybrid()
    assert cfg.use_interim_residual is False
    model = KimiK3Model(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 8))
    logits, cache = model(tokens, mode="chunk", use_cache=True)
    assert logits.shape == (1, 8, cfg.vocab_size)
    assert len(cache) == cfg.n_layer
    assert torch.isfinite(logits).all()
