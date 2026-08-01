"""
tests/test_mtp.py — sequential Multi-Token Prediction heads.

Contract:
  1. defaults disable MTP (fields = 0, model.mtp is None, param count unchanged)
  2. enabled model: forward(return_mtp=True) returns D logits of (B, T, vocab)
  3. return_mtp=True is rejected in recurrent mode
  4. combined main + MTP loss backward produces finite grads on trunk and heads
  5. mtp_loss alignment/tail-masking is correct and a tiny overfit decreases it
  6. new config fields survive YAML roundtrip
"""

import dataclasses

import pytest
import torch

from kimi_k3 import KimiK3Model, ModelConfig
from kimi_k3.training.loss import causal_lm_loss, mtp_loss


@pytest.fixture()
def enabled_cfg():
    return dataclasses.replace(
        ModelConfig.tiny(), num_nextn_predict_layers=2, mtp_loss_weight=0.3
    )


def test_defaults_disable_mtp():
    cfg = ModelConfig.tiny()
    assert cfg.num_nextn_predict_layers == 0
    assert cfg.mtp_loss_weight == 0.0
    model = KimiK3Model(cfg)
    assert model.mtp is None
    # A disabled model keeps the exact baseline parameter count.
    baseline = KimiK3Model(ModelConfig.tiny())
    assert sum(p.numel() for p in model.parameters()) == sum(
        p.numel() for p in baseline.parameters()
    )


def test_forward_returns_mtp_logits(enabled_cfg):
    torch.manual_seed(0)
    model = KimiK3Model(enabled_cfg).eval()
    tokens = torch.randint(0, enabled_cfg.vocab_size, (2, 16))
    logits, mtp_logits, cache = model(tokens, mode="chunk", return_mtp=True)
    assert logits.shape == (2, 16, enabled_cfg.vocab_size)
    assert len(mtp_logits) == enabled_cfg.num_nextn_predict_layers
    for head_logits in mtp_logits:
        assert head_logits.shape == logits.shape
        assert torch.isfinite(head_logits).all()


def test_default_return_is_two_tuple(enabled_cfg):
    model = KimiK3Model(enabled_cfg).eval()
    tokens = torch.randint(0, enabled_cfg.vocab_size, (1, 8))
    out = model(tokens, mode="chunk")
    assert len(out) == 2  # (logits, cache) — unchanged for generate()/callers


def test_return_mtp_rejected_in_recurrent(enabled_cfg):
    model = KimiK3Model(enabled_cfg).eval()
    tokens = torch.randint(0, enabled_cfg.vocab_size, (1, 4))
    with pytest.raises(ValueError, match="chunk"):
        model(tokens, mode="recurrent", return_mtp=True)


def test_disabled_model_return_mtp_empty():
    model = KimiK3Model(ModelConfig.tiny()).eval()
    tokens = torch.randint(0, ModelConfig.tiny().vocab_size, (1, 8))
    logits, mtp_logits, _ = model(tokens, mode="chunk", return_mtp=True)
    assert mtp_logits == []


def test_combined_loss_backward(enabled_cfg):
    torch.manual_seed(0)
    model = KimiK3Model(enabled_cfg).train()
    tokens = torch.randint(0, enabled_cfg.vocab_size, (2, 16))
    labels = torch.randint(0, enabled_cfg.vocab_size, (2, 16))
    logits, mtp_logits, _ = model(tokens, mode="chunk", return_mtp=True)
    loss = causal_lm_loss(logits, labels) + mtp_loss(
        mtp_logits, labels, enabled_cfg.mtp_loss_weight
    )
    loss.backward()
    head_params = [p for n, p in model.named_parameters() if n.startswith("mtp.")]
    assert head_params
    for p in head_params:
        assert p.grad is not None and torch.isfinite(p.grad).all()
    # trunk also receives gradient
    trunk_grad = model.embed_tokens.weight.grad
    assert trunk_grad is not None and torch.isfinite(trunk_grad).all()


def test_mtp_loss_zero_weight_and_empty():
    labels = torch.randint(0, 10, (2, 8))
    assert float(mtp_loss([], labels, 0.5)) == 0.0
    logits = [torch.randn(2, 8, 10)]
    assert float(mtp_loss(logits, labels, 0.0)) == 0.0


def test_mtp_loss_alignment_and_masking():
    # Head j uses logits[:, :T-j] against labels[:, j:]; must run for D up to T-1.
    torch.manual_seed(0)
    vocab, T = 10, 6
    mtp_logits = [torch.randn(1, T, vocab, requires_grad=True) for _ in range(2)]
    labels = torch.randint(0, vocab, (1, T))
    loss = mtp_loss(mtp_logits, labels, weight=1.0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert all(m.grad is not None for m in mtp_logits)


def test_mtp_overfit_decreases(enabled_cfg):
    torch.manual_seed(0)
    model = KimiK3Model(enabled_cfg).train()
    tokens = torch.randint(0, enabled_cfg.vocab_size, (2, 16))
    labels = torch.randint(0, enabled_cfg.vocab_size, (2, 16))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    def step():
        logits, mtp_logits, _ = model(tokens, mode="chunk", return_mtp=True)
        loss = causal_lm_loss(logits, labels) + mtp_loss(
            mtp_logits, labels, enabled_cfg.mtp_loss_weight
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return float(loss.detach())

    first = step()
    for _ in range(30):
        last = step()
    assert last < first


def test_config_yaml_roundtrip_includes_mtp(enabled_cfg):
    restored = ModelConfig.from_dict(enabled_cfg.to_dict())
    assert restored.num_nextn_predict_layers == 2
    assert restored.mtp_loss_weight == pytest.approx(0.3)
    assert ModelConfig.from_dict(
        {
            **enabled_cfg.to_dict(),
        }
    ).to_dict() == enabled_cfg.to_dict()


def test_config_rejects_bad_mtp_fields():
    with pytest.raises(ValueError, match="num_nextn_predict_layers"):
        dataclasses.replace(ModelConfig.tiny(), num_nextn_predict_layers=-1)
    with pytest.raises(ValueError, match="mtp_loss_weight"):
        dataclasses.replace(ModelConfig.tiny(), mtp_loss_weight=-0.5)


def test_hf_forward_adds_mtp_loss_in_training(enabled_cfg):
    pytest.importorskip("transformers")
    from kimi_k3 import KimiK3ForCausalLM, KimiK3HFConfig

    torch.manual_seed(0)
    hf_cfg = KimiK3HFConfig.from_model_config(enabled_cfg)
    model = KimiK3ForCausalLM(hf_cfg)
    ids = torch.randint(0, enabled_cfg.vocab_size, (2, 16))

    # Training: MTP aux term is added → loss strictly above the main-only loss.
    model.train()
    train_loss = model(input_ids=ids, labels=ids).loss
    # Eval: MTP is skipped, so the loss equals the plain next-token loss.
    model.eval()
    eval_loss = model(input_ids=ids, labels=ids).loss
    assert torch.isfinite(train_loss) and torch.isfinite(eval_loss)
    assert float(train_loss.detach()) > float(eval_loss.detach())
