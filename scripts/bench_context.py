#!/usr/bin/env python3
"""scripts/bench_context.py — sanity-run a long-context forward pass.

Usage:
    python scripts/bench_context.py --seq 65536          # KDA-only, default
    python scripts/bench_context.py --seq 8192 --hybrid  # include MLA layers

Why KDA-only by default:
    KDA is *linear* attention with a fixed-size recurrent state, so it handles
    64K/256K on CPU (memory is O(1) in sequence length; compute is O(T)). The
    Gated MLA layers are full softmax attention — O(T^2) memory — so a hybrid
    64K forward needs the production kernels / a GPU. Pass --hybrid only at
    modest --seq to feel that quadratic cost. This is the whole point of the
    KDA:MLA hybrid: KDA carries the long context cheaply.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import torch

from kimi_k3 import KimiK3Model, ModelConfig

PRESETS = {
    "tiny": ModelConfig.tiny,
    "small": ModelConfig.small,
    "kimi_1b_64k": ModelConfig.kimi_1b_64k,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="small", choices=sorted(PRESETS))
    ap.add_argument("--seq", type=int, default=65_536)
    ap.add_argument("--mode", default="chunk", choices=["chunk", "recurrent"])
    ap.add_argument(
        "--hybrid",
        action="store_true",
        help="keep MLA layers (O(T^2) memory) instead of forcing KDA-only",
    )
    args = ap.parse_args()

    base = PRESETS[args.preset]()
    cfg = replace(
        base,
        force_all_kda=not args.hybrid,
        max_seq_len=max(args.seq, base.max_seq_len),
        # keep it light so long sequences fit on CPU
        n_layer=min(base.n_layer, 4),
        use_interim_ffn=True,
    )
    torch.manual_seed(0)
    model = KimiK3Model(cfg).eval()

    tokens = torch.randint(0, cfg.vocab_size, (1, args.seq))
    kind = "hybrid (KDA+MLA)" if args.hybrid else "KDA-only (linear)"
    print(f"preset={args.preset}  seq={args.seq}  mode={args.mode}  attn={kind}  layers={cfg.n_layer}")

    t0 = time.time()
    with torch.no_grad():
        logits, _ = model(tokens, mode=args.mode)
    dt = time.time() - t0

    assert logits.shape == (1, args.seq, cfg.vocab_size)
    assert torch.isfinite(logits).all(), "non-finite logits — numerical blow-up"
    print(f"forward OK in {dt:.2f}s — logits {tuple(logits.shape)}, all finite")


if __name__ == "__main__":
    main()
