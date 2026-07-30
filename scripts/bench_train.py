#!/usr/bin/env python3
"""scripts/bench_train.py — measure training throughput (tokens/sec).

Times forward+backward for a preset on a device, so you can compare CPU vs MPS
and see the effect of the vectorized KDA ops.

    python scripts/bench_train.py --preset small --device cpu
    python scripts/bench_train.py --preset small --device mps      # on a Mac
    python scripts/bench_train.py --preset small --device cpu --reference   # loop KDA

`--reference` swaps in the transparent loop-based KDA (kda_chunkwise_reference)
so you can A/B the vectorized default against it on the same machine.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import torch

from kimi_k3 import KimiK3Model, ModelConfig
from kimi_k3.training import resolve_device

PRESETS = {
    "tiny": ModelConfig.tiny,
    "tiny_hybrid": ModelConfig.tiny_hybrid,
    "small": ModelConfig.small,
    "kimi_1b_64k": ModelConfig.kimi_1b_64k,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="small", choices=sorted(PRESETS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--reference", action="store_true", help="use loop-based KDA")
    args = ap.parse_args()

    if args.reference:
        # Route the KDA layer through the transparent loop version.
        import kimi_k3.attention.kda as kda_mod
        from kimi_k3.attention.kda_ops import kda_chunkwise_reference

        kda_mod.kda_chunkwise = kda_chunkwise_reference

    device = resolve_device(args.device)
    torch.manual_seed(0)
    cfg = replace(PRESETS[args.preset](), vocab_size=259, max_seq_len=max(args.seq_len, 64))
    model = KimiK3Model(cfg).to(device).train()
    x = torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len), device=device)

    def one_step() -> None:
        model.zero_grad(set_to_none=True)
        logits, _ = model(x, mode="chunk")
        logits.float().sum().backward()

    for _ in range(2):  # warmup
        one_step()
    t0 = time.time()
    for _ in range(args.steps):
        one_step()
    dt = (time.time() - t0) / args.steps

    toks = args.batch_size * args.seq_len
    impl = "reference(loop)" if args.reference else "vectorized"
    print(
        f"{args.preset} | {impl} | device={device.type} | "
        f"batch{args.batch_size}×seq{args.seq_len} | {dt * 1000:.0f} ms/step | "
        f"{toks / dt:,.0f} tok/s"
    )


if __name__ == "__main__":
    main()
