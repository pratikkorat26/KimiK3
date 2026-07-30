"""
tests/test_kda.py — numerical verification of KimiDeltaAttention.

Contract:
  1. output / KDACache shapes
  2. chunkwise ≈ recurrent
  3. cached prefill + decode ≈ full recurrent
  4. backward through chunkwise
"""

import pytest
import torch

from kimi_k3 import KDACache, KDAConfig, KimiDeltaAttention, MLACache

TOL = 1e-5


@pytest.fixture()
def cfg():
    return KDAConfig(hidden_size=256, num_heads=2, head_dim_k=64, head_dim_v=64, chunk_size=16)


@pytest.fixture()
def layer(cfg):
    torch.manual_seed(0)
    return KimiDeltaAttention(cfg).eval()


def test_output_shapes(cfg, layer):
    x = torch.randn(1, 37, cfg.hidden_size)
    out, cache = layer(x, mode="chunk", use_cache=True)

    assert out.shape == (1, 37, cfg.hidden_size)
    assert isinstance(cache, KDACache)
    assert cache.recurrent.shape == (1, cfg.num_heads, cfg.head_dim_k, cfg.head_dim_v)
    assert cache.conv_q.shape == (1, cfg.conv_kernel_size - 1, cfg.num_heads * cfg.head_dim_k)
    assert cache.conv_v.shape == (1, cfg.conv_kernel_size - 1, cfg.num_heads * cfg.head_dim_v)


def test_chunk_matches_recurrent(cfg, layer):
    x = torch.randn(1, 37, cfg.hidden_size)
    with torch.no_grad():
        out_chunk, _ = layer(x, mode="chunk")
        out_recur, _ = layer(x, mode="recurrent")

    assert (out_chunk - out_recur).abs().max().item() < TOL


def test_cache_continuation(cfg, layer):
    x = torch.randn(1, 37, cfg.hidden_size)
    with torch.no_grad():
        ref, _ = layer(x, mode="recurrent")
        pre, cache = layer(x[:, :20], mode="chunk", use_cache=True)
        outs = []
        for t in range(20, 37):
            o, cache = layer(x[:, t : t + 1], mode="recurrent", cache=cache, use_cache=True)
            outs.append(o)
        cont = torch.cat([pre] + outs, dim=1)

    assert (ref - cont).abs().max().item() < TOL


@pytest.mark.parametrize("mode", ["chunk", "recurrent"])
def test_cache_multi_token_continuation(cfg, layer, mode):
    x = torch.randn(1, 37, cfg.hidden_size)
    with torch.no_grad():
        ref, _ = layer(x, mode="recurrent")
        _, cache = layer(x[:, :20], mode="chunk", use_cache=True)
        tail, new_cache = layer(
            x[:, 20:],
            mode=mode,
            cache=cache,
            use_cache=True,
        )

    assert torch.allclose(ref[:, 20:], tail, atol=TOL, rtol=TOL)
    assert new_cache.recurrent.shape == cache.recurrent.shape


def test_backward_runs(cfg, layer):
    layer.train()
    x = torch.randn(1, 37, cfg.hidden_size, requires_grad=True)
    out, _ = layer(x, mode="chunk")
    out.sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def _kda_inputs(B, H, T, d_k, d_v, seed=0):
    """Realistic KDA inputs: q/k L2-normalized (as the layer produces them)."""
    import torch.nn.functional as F

    torch.manual_seed(seed)
    q = F.normalize(torch.randn(B, H, T, d_k), dim=-1)
    k = F.normalize(torch.randn(B, H, T, d_k), dim=-1)
    v = torch.randn(B, H, T, d_v)
    g = -torch.rand(B, H, T, d_k) * 2.0            # per-channel log-decay <= 0
    beta = torch.sigmoid(torch.randn(B, H, T))
    return q, k, v, g, beta


@pytest.mark.parametrize("T", [16, 40, 37])
def test_vectorized_chunkwise_matches_reference(T):
    from kimi_k3.attention.kda_ops import kda_chunkwise, kda_chunkwise_reference

    q, k, v, g, beta = _kda_inputs(2, 3, T, 16, 16)
    o_v, S_v = kda_chunkwise(q, k, v, g, beta, 8)
    o_r, S_r = kda_chunkwise_reference(q, k, v, g, beta, 8)
    assert torch.allclose(o_v, o_r, atol=1e-4, rtol=1e-4)
    assert torch.allclose(S_v, S_r, atol=1e-4, rtol=1e-4)


def test_vectorized_chunkwise_matches_reference_state_and_grad():
    from kimi_k3.attention.kda_ops import kda_chunkwise, kda_chunkwise_reference

    q, k, v, g, beta = _kda_inputs(2, 2, 24, 16, 16)
    S0 = torch.randn(2, 2, 16, 16)
    qv = q.clone().requires_grad_(True)
    qr = q.clone().requires_grad_(True)
    o_v, _ = kda_chunkwise(qv, k, v, g, beta, 8, S0)
    o_r, _ = kda_chunkwise_reference(qr, k, v, g, beta, 8, S0)
    assert torch.allclose(o_v, o_r, atol=1e-4, rtol=1e-4)
    o_v.sum().backward()
    o_r.sum().backward()
    assert torch.allclose(qv.grad, qr.grad, atol=1e-4, rtol=1e-4)


def test_layer_rejects_wrong_cache_type(cfg, layer):
    cache = MLACache(
        kv_latent=torch.zeros(1, 1, 4),
        k_shared=torch.zeros(1, 1, 4),
    )
    with pytest.raises(TypeError, match="KDACache"):
        layer(torch.randn(1, 1, cfg.hidden_size), cache=cache)
