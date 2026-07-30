"""Boundary validation, configuration validation, and generation contracts."""

from dataclasses import replace

import pytest
import torch

from kimi_k3 import (
    AttentionCache,
    KDACache,
    KimiK3Model,
    MLACache,
    ModelConfig,
)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("vocab_size", 0),
        ("hidden_size", 0),
        ("n_layer", 0),
        ("conv_kernel_size", 0),
        ("chunk_size", 0),
        ("attn_res_block_size", 0),
        ("n_experts", 0),
        ("n_shared_experts", 0),
        ("eps", 0.0),
        ("situ_beta_gate", 0.0),
    ],
)
def test_config_rejects_non_positive_values(field, value):
    with pytest.raises(ValueError, match=field):
        ModelConfig(**{field: value})


def test_config_rejects_invalid_expert_relationship():
    with pytest.raises(ValueError, match="n_experts_per_tok"):
        ModelConfig(n_experts=2, n_experts_per_tok=3)


def test_config_rejects_invalid_dense_layer_count():
    with pytest.raises(ValueError, match="n_dense_layers"):
        ModelConfig(n_layer=2, n_dense_layers=3)


def test_config_allows_zero_dense_layers():
    cfg = ModelConfig.tiny()
    assert cfg.n_dense_layers == 0


def make_tiny_model(*, hybrid=False):
    cfg = ModelConfig.tiny_hybrid() if hybrid else ModelConfig.tiny()
    torch.manual_seed(0)
    return cfg, KimiK3Model(cfg)


@pytest.mark.parametrize(
    "tokens",
    [
        torch.zeros(2, 3, 4, dtype=torch.long),
        torch.zeros(2, 3, dtype=torch.float32),
        torch.zeros(0, 3, dtype=torch.long),
        torch.zeros(2, 0, dtype=torch.long),
    ],
)
def test_forward_rejects_invalid_token_shape_or_dtype(tokens):
    _, model = make_tiny_model()
    with pytest.raises((TypeError, ValueError)):
        model(tokens)


@pytest.mark.parametrize("token_id", [-1, 4096])
def test_forward_rejects_out_of_range_tokens(token_id):
    _, model = make_tiny_model()
    tokens = torch.tensor([[token_id]], dtype=torch.long)
    with pytest.raises(ValueError, match="token ids"):
        model(tokens)


def test_forward_rejects_wrong_cache_layer_count():
    cfg, model = make_tiny_model()
    tokens = torch.zeros(1, 1, dtype=torch.long)
    cache = AttentionCache.empty(cfg.n_layer - 1)
    with pytest.raises(ValueError, match="layers"):
        model(tokens, cache=cache)


def test_forward_rejects_wrong_cache_layer_type():
    cfg, model = make_tiny_model()
    tokens = torch.zeros(1, 2, dtype=torch.long)
    _, cache = model(tokens, use_cache=True)
    cache.layers[0] = MLACache(
        kv_latent=torch.zeros(1, 2, cfg.kv_latent_dim),
        k_shared=torch.zeros(1, 2, cfg.qk_shared_dim),
    )

    with pytest.raises(TypeError, match="KDACache"):
        model(tokens[:, :1], cache=cache)


def test_forward_rejects_cache_batch_mismatch():
    _, model = make_tiny_model()
    prefix = torch.zeros(1, 2, dtype=torch.long)
    _, cache = model(prefix, use_cache=True)
    continuation = torch.zeros(2, 1, dtype=torch.long)

    with pytest.raises(ValueError, match="shape"):
        model(continuation, cache=cache)


def test_forward_rejects_cache_shape_mismatch():
    _, model = make_tiny_model()
    tokens = torch.zeros(1, 2, dtype=torch.long)
    _, cache = model(tokens, use_cache=True)
    layer_cache = cache.get(0)
    assert isinstance(layer_cache, KDACache)
    layer_cache.conv_q = layer_cache.conv_q[:, :-1]

    with pytest.raises(ValueError, match="conv_q"):
        model(tokens[:, :1], cache=cache)


def test_forward_rejects_cache_dtype_mismatch():
    _, model = make_tiny_model()
    tokens = torch.zeros(1, 2, dtype=torch.long)
    _, cache = model(tokens, use_cache=True)
    layer_cache = cache.get(0)
    assert isinstance(layer_cache, KDACache)
    layer_cache.conv_q = layer_cache.conv_q.double()
    layer_cache.conv_k = layer_cache.conv_k.double()
    layer_cache.conv_v = layer_cache.conv_v.double()

    with pytest.raises(TypeError, match="dtype"):
        model(tokens[:, :1], cache=cache)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_forward_rejects_cache_device_mismatch():
    _, model = make_tiny_model()
    tokens = torch.zeros(1, 2, dtype=torch.long)
    _, cache = model(tokens, use_cache=True)
    layer_cache = cache.get(0)
    assert isinstance(layer_cache, KDACache)
    layer_cache.conv_q = layer_cache.conv_q.cuda()

    with pytest.raises(ValueError, match="device"):
        model(tokens[:, :1], cache=cache)


def test_forward_rejects_inconsistent_mla_cache_length():
    _, model = make_tiny_model(hybrid=True)
    tokens = torch.zeros(1, 2, dtype=torch.long)
    _, cache = model(tokens, use_cache=True)
    cache.tokens_seen = 3

    with pytest.raises(ValueError, match="tokens_seen"):
        model(tokens[:, :1], cache=cache)


def test_forward_enforces_context_limit_with_cache():
    cfg = replace(ModelConfig.tiny(), max_seq_len=3)
    model = KimiK3Model(cfg)
    tokens = torch.zeros(1, 3, dtype=torch.long)
    _, cache = model(tokens, use_cache=True)

    with pytest.raises(ValueError, match="max_seq_len"):
        model(tokens[:, :1], cache=cache)


def test_generate_zero_returns_input_unchanged():
    _, model = make_tiny_model()
    tokens = torch.tensor([[1, 2]], dtype=torch.long)
    result = model.generate(tokens, 0)
    assert result is tokens


def test_generate_one_appends_one_token_and_restores_training_state():
    _, model = make_tiny_model()
    model.train()
    tokens = torch.tensor([[1, 2]], dtype=torch.long)
    result = model.generate(tokens, 1)

    assert result.shape == (1, 3)
    assert model.training


def test_generate_preserves_eval_state():
    _, model = make_tiny_model()
    model.eval()
    tokens = torch.tensor([[1, 2]], dtype=torch.long)
    model.generate(tokens, 1)
    assert not model.training


def test_generate_restores_state_when_forward_raises(monkeypatch):
    _, model = make_tiny_model()
    model.train()
    tokens = torch.tensor([[1, 2]], dtype=torch.long)

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(model, "forward", fail)
    with pytest.raises(RuntimeError, match="synthetic"):
        model.generate(tokens, 1)
    assert model.training


@pytest.mark.parametrize("max_new_tokens", [-1, 1.5, True])
def test_generate_rejects_invalid_token_count(max_new_tokens):
    _, model = make_tiny_model()
    tokens = torch.tensor([[1, 2]], dtype=torch.long)
    with pytest.raises((TypeError, ValueError), match="max_new_tokens"):
        model.generate(tokens, max_new_tokens)


def test_generate_rejects_empty_prompt():
    _, model = make_tiny_model()
    tokens = torch.empty(1, 0, dtype=torch.long)
    with pytest.raises(ValueError, match="sequence dimension"):
        model.generate(tokens, 1)


def test_generate_rejects_context_overflow():
    cfg = replace(ModelConfig.tiny(), max_seq_len=2)
    model = KimiK3Model(cfg)
    tokens = torch.tensor([[1, 2]], dtype=torch.long)
    with pytest.raises(ValueError, match="max_seq_len"):
        model.generate(tokens, 1)


def test_generate_sampling_runs_and_stays_in_vocab():
    cfg, model = make_tiny_model()
    model.eval()
    tokens = torch.tensor([[1, 2]], dtype=torch.long)
    gen = torch.Generator().manual_seed(0)
    out = model.generate(
        tokens, 4, do_sample=True, temperature=0.8, top_k=5, top_p=0.9, generator=gen
    )
    assert out.shape == (1, 6)
    assert out.min() >= 0 and out.max() < cfg.vocab_size


def test_generate_sampling_is_reproducible_with_generator():
    _, model = make_tiny_model()
    model.eval()
    tokens = torch.tensor([[1, 2]], dtype=torch.long)
    a = model.generate(tokens, 3, do_sample=True, generator=torch.Generator().manual_seed(7))
    b = model.generate(tokens, 3, do_sample=True, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


def test_generate_greedy_matches_default():
    _, model = make_tiny_model()
    model.eval()
    tokens = torch.tensor([[1, 2]], dtype=torch.long)
    default = model.generate(tokens, 3)
    explicit_greedy = model.generate(tokens, 3, do_sample=False)
    assert torch.equal(default, explicit_greedy)


def test_generate_eos_stops_early():
    _, model = make_tiny_model()
    model.eval()
    tokens = torch.tensor([[1, 2]], dtype=torch.long)
    # Force EOS to be whatever greedy picks first, so decoding halts immediately.
    logits, _ = model(tokens, mode="chunk", use_cache=True)
    first = int(logits[:, -1].argmax(-1).item())
    out = model.generate(tokens, 10, eos_token_id=first)
    # prompt (2) + the single emitted EOS token
    assert out.shape[1] == 3
    assert int(out[0, -1].item()) == first


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (dict(do_sample=True, temperature=0.0), "temperature"),
        (dict(do_sample=True, top_k=0), "top_k"),
        (dict(do_sample=True, top_p=1.5), "top_p"),
    ],
)
def test_generate_rejects_invalid_sampling_params(kwargs, match):
    _, model = make_tiny_model()
    tokens = torch.tensor([[1, 2]], dtype=torch.long)
    with pytest.raises((TypeError, ValueError), match=match):
        model.generate(tokens, 2, **kwargs)
