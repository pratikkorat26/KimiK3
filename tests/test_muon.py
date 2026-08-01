"""
tests/test_muon.py — Per-Head Muon optimizer (hybrid Muon + AdamW).

Contract:
  1. Newton-Schulz pushes singular values into a bounded band (semi-orthogonal)
  2. batched Newton-Schulz == looping over the batch (per-head equivalence)
  3. collect_muon_param_groups partitions params exactly once, disjointly
  4. embeddings / lm_head / router / 1-D / conv stay on AdamW; attention
     projections carry per-head splits
  5. a Muon step runs and keeps params finite; head-split reshape is correct
  6. optimizer="muon" overfits a fixed batch; optimizer="adamw" is unchanged
"""

import dataclasses

import pytest
import torch

from kimi_k3 import KimiK3Model, ModelConfig
from kimi_k3.training import TrainConfig, Trainer
from kimi_k3.training.muon import (
    Muon,
    build_muon_optimizer,
    collect_muon_param_groups,
    zeropower_via_newtonschulz5,
)


@pytest.mark.parametrize("shape", [(64, 16), (16, 64), (32, 32)])
def test_newtonschulz_bounds_singular_values(shape):
    torch.manual_seed(0)
    g = torch.randn(*shape)
    o = zeropower_via_newtonschulz5(g, steps=5)
    assert o.shape == g.shape
    sv = torch.linalg.svdvals(o)
    # Muon's 5-step NS lands singular values in a bounded band around 1 (not
    # exactly 1 by design), and never blows up.
    assert float(sv.max()) < 1.6
    assert float(sv.min()) > 0.4
    assert torch.isfinite(o).all()


def test_newtonschulz_batched_equals_loop():
    torch.manual_seed(0)
    batch = torch.randn(4, 20, 12)
    batched = zeropower_via_newtonschulz5(batch, steps=5)
    looped = torch.stack([zeropower_via_newtonschulz5(batch[i], steps=5) for i in range(4)])
    assert torch.allclose(batched, looped, atol=1e-5)


def test_partition_is_exact_and_disjoint():
    model = KimiK3Model(ModelConfig.small())
    specs, adamw = collect_muon_param_groups(model)
    muon_ids = {id(s["param"]) for s in specs}
    adamw_ids = {id(p) for p in adamw}
    all_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert muon_ids.isdisjoint(adamw_ids)
    assert muon_ids | adamw_ids == all_ids
    total_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_split = sum(s["param"].numel() for s in specs) + sum(p.numel() for p in adamw)
    assert total_split == total_model


def test_classification_excludes_embed_head_router():
    model = KimiK3Model(ModelConfig.small())
    specs, adamw = collect_muon_param_groups(model)
    names = {id(p): n for n, p in model.named_parameters()}
    for spec in specs:
        name = names[id(spec["param"])]
        assert "embed_tokens" not in name
        assert "lm_head" not in name
        assert "router" not in name
        assert spec["param"].ndim == 2  # Muon only orthogonalizes matrices
    adamw_names = [names[id(p)] for p in adamw]
    assert any("embed_tokens" in n for n in adamw_names)
    assert any("router" in n for n in adamw_names)
    # depthwise conv weights (3-D) and 1-D scales must be on AdamW
    assert all(p.ndim != 2 or "embed" in names[id(p)] or "lm_head" in names[id(p)]
               or "router" in names[id(p)] for p in adamw)
    # attention projections carry a per-head split
    head_params = [s for s in specs if s["head_split"] is not None]
    assert head_params
    assert any("attn.w_q" in names[id(s["param"])] for s in head_params)


def test_muon_step_runs_and_is_finite():
    torch.manual_seed(0)
    model = KimiK3Model(ModelConfig.small())
    opt = build_muon_optimizer(model, muon_lr=0.02, adam_lr=3e-4)
    assert isinstance(opt, Muon)
    tokens = torch.randint(0, ModelConfig.small().vocab_size, (2, 32))
    logits, _ = model(tokens, mode="chunk")
    logits.float().pow(2).mean().backward()
    before = model.blocks[0].attn.w_q.weight.detach().clone()
    opt.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    # a per-head projection actually moved
    assert not torch.equal(before, model.blocks[0].attn.w_q.weight)


def test_muon_overfits_fixed_batch():
    torch.manual_seed(0)
    cfg = dataclasses.replace(ModelConfig.tiny_hybrid(), vocab_size=64, max_seq_len=64)
    model = KimiK3Model(cfg)
    x = torch.randint(0, cfg.vocab_size, (4, 32))
    y = torch.randint(0, cfg.vocab_size, (4, 32))
    tc = TrainConfig(
        max_steps=120, warmup_steps=12, lr=3e-3, device="cpu", optimizer="muon"
    )
    trainer = Trainer(model, tc)
    losses = trainer.overfit(x, y, 120)
    assert losses[-1] < 0.2 * losses[0]


def test_adamw_default_is_plain_adamw():
    model = KimiK3Model(ModelConfig.tiny())
    trainer = Trainer(model, TrainConfig(device="cpu"))
    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    assert not isinstance(trainer.optimizer, Muon)
    # initial_lr stamped for the proportional schedule
    assert all("initial_lr" in g for g in trainer.optimizer.param_groups)


def test_lr_schedule_scales_groups_proportionally():
    torch.manual_seed(0)
    model = KimiK3Model(ModelConfig.tiny())
    tc = TrainConfig(max_steps=100, warmup_steps=10, lr=3e-4, device="cpu", optimizer="muon")
    trainer = Trainer(model, tc)
    trainer._apply_lr(5)  # mid-warmup: multiplier = 6/10
    for group in trainer.optimizer.param_groups:
        assert group["lr"] == pytest.approx(group["initial_lr"] * 0.6)


def test_config_rejects_bad_muon_fields():
    with pytest.raises(ValueError, match="optimizer"):
        TrainConfig(optimizer="sgd")
    with pytest.raises(ValueError, match="muon_momentum"):
        TrainConfig(muon_momentum=1.0)
    with pytest.raises(ValueError, match="muon_ns_steps"):
        TrainConfig(muon_ns_steps=0)
