#!/usr/bin/env python3
"""scripts/overfit.py — the sanity gate: can the model memorize one batch?

Trains tiny_hybrid on a single fixed batch and prints the loss collapsing toward
zero. If this does NOT drop, something in the forward/backward/optimizer path is
broken — it is the fastest end-to-end correctness check.

    python scripts/overfit.py [steps]
"""

from __future__ import annotations

import sys
from dataclasses import replace

import torch

from kimi_k3 import KimiK3Model, ModelConfig
from kimi_k3.data import TINY_CORPUS, batch_iterator, build_dataset
from kimi_k3.tokenizer import ByteTokenizer
from kimi_k3.training import TrainConfig, Trainer


def main() -> None:
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    torch.manual_seed(0)

    tok = ByteTokenizer()
    seq_len = 32
    cfg = replace(ModelConfig.tiny_hybrid(), vocab_size=tok.vocab_size, max_seq_len=64)
    model = KimiK3Model(cfg)

    ds = build_dataset(TINY_CORPUS, tok, seq_len=seq_len)
    x, y = next(batch_iterator(ds, batch_size=4, shuffle=False))

    tc = TrainConfig(max_steps=steps, warmup_steps=max(1, steps // 10), lr=3e-3, device="cpu")
    trainer = Trainer(model, tc)

    losses = trainer.overfit(x, y, steps)
    for i in range(0, steps, max(1, steps // 10)):
        print(f"step {i:4d} | loss {losses[i]:.4f}")
    print(f"final    | loss {losses[-1]:.4f}   (start {losses[0]:.4f})")
    ok = losses[-1] < 0.5 * losses[0]
    print("OK — model memorized the batch" if ok else "WARNING — loss did not collapse")


if __name__ == "__main__":
    main()
