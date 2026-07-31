"""Hugging Face model-contract and activation-checkpointing tests."""

from dataclasses import replace

import pytest
import torch

transformers = pytest.importorskip("transformers")

from kimi_k3 import KimiK3Model, ModelConfig  # noqa: E402
from kimi_k3.hf import KimiK3ForCausalLM, KimiK3HFConfig  # noqa: E402


def _hf_tiny(vocab_size: int = 64) -> KimiK3ForCausalLM:
    config = replace(
        ModelConfig.tiny_hybrid(),
        vocab_size=vocab_size,
        max_seq_len=64,
    )
    return KimiK3ForCausalLM(
        KimiK3HFConfig.from_model_config(
            config,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )


def test_hf_forward_shifts_labels_and_returns_loss():
    torch.manual_seed(0)
    model = _hf_tiny()
    input_ids = torch.randint(3, 64, (2, 12))
    output = model(input_ids=input_ids, labels=input_ids)

    expected = torch.nn.functional.cross_entropy(
        output.logits[:, :-1].reshape(-1, 64),
        input_ids[:, 1:].reshape(-1),
    )
    assert output.loss is not None
    assert torch.allclose(output.loss, expected)


def test_hf_rejects_padding_mask():
    model = _hf_tiny()
    input_ids = torch.randint(3, 64, (1, 8))
    mask = torch.ones_like(input_ids)
    mask[:, -1] = 0
    with pytest.raises(ValueError, match="packed"):
        model(input_ids=input_ids, attention_mask=mask)


def test_hf_gradient_checkpointing_hook():
    model = _hf_tiny()
    model.gradient_checkpointing_enable()
    assert model.model.gradient_checkpointing
    model.gradient_checkpointing_disable()
    assert not model.model.gradient_checkpointing


def test_hf_save_load_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = _hf_tiny().eval()
    input_ids = torch.randint(3, 64, (1, 8))
    before = model(input_ids=input_ids).logits
    model.save_pretrained(tmp_path, safe_serialization=True)

    loaded = KimiK3ForCausalLM.from_pretrained(tmp_path).eval()
    after = loaded(input_ids=input_ids).logits
    assert torch.equal(before, after)


def test_gradient_checkpointing_matches_regular_backward():
    torch.manual_seed(0)
    config = replace(ModelConfig.tiny_hybrid(), vocab_size=64, max_seq_len=64)
    regular = KimiK3Model(config).train()
    checkpointed = KimiK3Model(config).train()
    checkpointed.load_state_dict(regular.state_dict())
    checkpointed.gradient_checkpointing_enable()
    input_ids = torch.randint(0, config.vocab_size, (1, 16))

    regular_logits, _ = regular(input_ids)
    checkpointed_logits, _ = checkpointed(input_ids)
    assert torch.allclose(regular_logits, checkpointed_logits)

    regular_logits.float().sum().backward()
    checkpointed_logits.float().sum().backward()
    for regular_parameter, checkpointed_parameter in zip(
        regular.parameters(),
        checkpointed.parameters(),
    ):
        assert (regular_parameter.grad is None) == (
            checkpointed_parameter.grad is None
        )
        if regular_parameter.grad is None:
            continue
        assert torch.allclose(
            regular_parameter.grad,
            checkpointed_parameter.grad,
            atol=1e-5,
            rtol=1e-4,
        )
