#!/usr/bin/env python3
"""scripts/param_count.py — report total & active parameter counts for a preset.

Usage:
    python scripts/param_count.py [preset]        # default: kimi_1b_64k
    python scripts/param_count.py small

"Active" params = the subset used for a single token: everything except the
routed experts that a token does NOT select (top-k of n_experts per MoE layer).
"""

from __future__ import annotations

import sys

from kimi_k3 import KimiK3Model, ModelConfig
from kimi_k3.moe.latent_moe import StableLatentMoE

PRESETS = {
    "tiny": ModelConfig.tiny,
    "tiny_hybrid": ModelConfig.tiny_hybrid,
    "small": ModelConfig.small,
    "kimi_1b_64k": ModelConfig.kimi_1b_64k,
    "full": ModelConfig,
}


def _human(n: float) -> str:
    for unit, scale in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= scale:
            return f"{n / scale:.2f}{unit}"
    return str(int(n))


def active_params(model: KimiK3Model, cfg: ModelConfig) -> int:
    """Total params minus the routed experts not selected per token."""
    total = sum(p.numel() for p in model.parameters())
    inactive = 0
    for module in model.modules():
        if isinstance(module, StableLatentMoE):
            per_expert = sum(p.numel() for p in module.routed_experts[0].parameters())
            skipped = cfg.n_experts - cfg.n_experts_per_tok
            inactive += per_expert * skipped
    return total - inactive


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "kimi_1b_64k"
    if name not in PRESETS:
        raise SystemExit(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    cfg = PRESETS[name]()
    model = KimiK3Model(cfg)

    total = sum(p.numel() for p in model.parameters())
    embed = model.embed_tokens.weight.numel()
    head = 0 if cfg.tie_word_embeddings else model.lm_head.weight.numel()
    active = active_params(model, cfg)

    print(f"preset               {name}")
    print(f"layers / hidden      {cfg.n_layer} / {cfg.hidden_size}")
    print(f"experts (active/tot) {cfg.n_experts_per_tok}/{cfg.n_experts}")
    print(f"vocab / ctx          {cfg.vocab_size} / {cfg.max_seq_len}")
    print("-" * 40)
    print(f"embeddings           {_human(embed + head)}")
    print(f"total params         {_human(total)}  ({total:,})")
    print(f"active params/token  {_human(active)}  ({active:,})")


if __name__ == "__main__":
    main()
