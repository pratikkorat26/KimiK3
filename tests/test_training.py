"""tests/test_training.py — loss, config, checkpointing, and the overfit gate.

The overfit test is the key correctness gate: if the model cannot memorize a
fixed batch, the forward/backward/optimizer path is broken.
"""

import math
from dataclasses import replace

import pytest
import torch

from kimi_k3 import KimiK3Model, ModelConfig
from kimi_k3.training import TrainConfig, Trainer, causal_lm_loss


def _tiny_model(vocab=64):
    torch.manual_seed(0)
    cfg = replace(ModelConfig.tiny(), vocab_size=vocab, max_seq_len=64)
    return KimiK3Model(cfg), cfg


def test_causal_lm_loss_uniform_is_log_vocab():
    vocab = 50
    logits = torch.zeros(2, 5, vocab)  # uniform → loss = log(vocab)
    labels = torch.randint(0, vocab, (2, 5))
    loss = causal_lm_loss(logits, labels)
    assert abs(float(loss) - math.log(vocab)) < 1e-4


def test_overfit_memorizes_batch():
    model, cfg = _tiny_model()
    tc = TrainConfig(max_steps=120, warmup_steps=10, lr=3e-3, device="cpu")
    trainer = Trainer(model, tc)

    torch.manual_seed(1)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    losses = trainer.overfit(x, y, 120)

    assert losses[-1] < 0.2 * losses[0], f"loss did not collapse: {losses[0]:.2f}→{losses[-1]:.2f}"
    assert losses[-1] < 0.5


def test_checkpoint_save_load_roundtrip(tmp_path):
    model, cfg = _tiny_model()
    tc = TrainConfig(max_steps=10, warmup_steps=2, ckpt_dir=str(tmp_path), device="cpu")
    trainer = Trainer(model, tc)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    trainer.overfit(x, y, 5)
    path = trainer.save_checkpoint(tag="t")

    model2, _ = _tiny_model()
    trainer2 = Trainer(model2, tc)
    trainer2.load_checkpoint(path)
    assert trainer2.step == trainer.step
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)


def test_train_config_rejects_bad_values():
    with pytest.raises(ValueError, match="batch_size"):
        TrainConfig(batch_size=0)
    with pytest.raises(ValueError, match="warmup_steps"):
        TrainConfig(max_steps=10, warmup_steps=20)
    with pytest.raises(ValueError, match="device"):
        TrainConfig(device="tpu")
