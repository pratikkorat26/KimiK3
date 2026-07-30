#!/usr/bin/env python3
"""scripts/smoke_forward.py — run a forward pass + greedy/sampled generate.

Usage:
    python scripts/smoke_forward.py [preset]      # default: small

Confirms the whole stack (embed → hybrid blocks → AttnRes → MoE → lm_head)
runs and that both decode paths produce finite, in-vocab tokens.
"""

from __future__ import annotations

import sys

import torch

from kimi_k3 import KimiK3Model, ModelConfig

PRESETS = {
    "tiny": ModelConfig.tiny,
    "tiny_hybrid": ModelConfig.tiny_hybrid,
    "small": ModelConfig.small,
    "kimi_1b_64k": ModelConfig.kimi_1b_64k,
}


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "small"
    if name not in PRESETS:
        raise SystemExit(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")

    torch.manual_seed(0)
    cfg = PRESETS[name]()
    model = KimiK3Model(cfg).eval()

    B, T = 1, 16
    tokens = torch.randint(0, cfg.vocab_size, (B, T))

    logits, cache = model(tokens, mode="chunk", use_cache=True)
    assert logits.shape == (B, T, cfg.vocab_size)
    assert torch.isfinite(logits).all()
    print(f"[{name}] forward: logits {tuple(logits.shape)}  cache layers={len(cache)}")

    greedy = model.generate(tokens, 8)
    assert greedy.shape == (B, T + 8)
    print(f"[{name}] greedy  generate → {tuple(greedy.shape)}")

    gen = torch.Generator().manual_seed(0)
    sampled = model.generate(
        tokens, 8, do_sample=True, temperature=0.8, top_k=20, top_p=0.95, generator=gen
    )
    assert sampled.shape == (B, T + 8)
    assert sampled.min() >= 0 and sampled.max() < cfg.vocab_size
    print(f"[{name}] sampled generate → {tuple(sampled.shape)}  (temp=0.8, top_k=20, top_p=0.95)")
    print("OK")


if __name__ == "__main__":
    main()
