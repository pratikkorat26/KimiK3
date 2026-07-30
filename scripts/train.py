#!/usr/bin/env python3
"""scripts/train.py — train a KimiK3 preset on a text corpus.

    python scripts/train.py --preset small --steps 200
    python scripts/train.py --preset small --data mytext.txt --tokenizer gpt2
    python scripts/train.py --resume out/ckpt_final.pt --steps 400

Defaults run offline on CPU over the bundled tiny corpus. Keep seq_len / batch
small on CPU — the readable ops are not throughput-optimized (see docs/roadmap.md).
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import torch

from kimi_k3 import KimiK3Model, ModelConfig
from kimi_k3.data import TINY_CORPUS, build_dataset, load_text
from kimi_k3.tokenizer import ByteTokenizer
from kimi_k3.training import TrainConfig, Trainer

PRESETS = {
    "tiny": ModelConfig.tiny,
    "tiny_hybrid": ModelConfig.tiny_hybrid,
    "small": ModelConfig.small,
    "kimi_1b_64k": ModelConfig.kimi_1b_64k,
}


def make_tokenizer(name: str):
    if name == "byte":
        return ByteTokenizer()
    if name == "gpt2":
        from kimi_k3.tokenizer import TiktokenTokenizer

        return TiktokenTokenizer("gpt2")
    raise SystemExit(f"unknown tokenizer {name!r} (choose byte|gpt2)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="small", choices=sorted(PRESETS))
    ap.add_argument("--tokenizer", default="byte", choices=["byte", "gpt2"])
    ap.add_argument("--data", default=None, help="path to a UTF-8 text file (default: bundled corpus)")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", default=None, help="checkpoint path to resume from")
    args = ap.parse_args()

    torch.manual_seed(0)
    tok = make_tokenizer(args.tokenizer)
    text = load_text(args.data) if args.data else TINY_CORPUS
    train_ds = build_dataset(text, tok, seq_len=args.seq_len)
    print(f"corpus: {len(text)} chars → {len(train_ds)} blocks of {args.seq_len}")

    cfg = replace(
        PRESETS[args.preset](),
        vocab_size=tok.vocab_size,
        max_seq_len=max(args.seq_len, 64),
    )
    model = KimiK3Model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {args.preset}  {n_params / 1e6:.1f}M params  vocab {cfg.vocab_size}")

    tc = TrainConfig(
        max_steps=args.steps,
        warmup_steps=max(1, args.steps // 10),
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        device=args.device,
    )
    trainer = Trainer(model, tc)
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"resumed from {args.resume} at step {trainer.step}")
    trainer.fit(train_ds, val_ds=train_ds)
    print("done.")


if __name__ == "__main__":
    main()
