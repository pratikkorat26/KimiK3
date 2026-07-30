#!/usr/bin/env python3
"""scripts/generate_text.py — generate text from a trained checkpoint.

Closes the loop: tokenize a prompt → model.generate (Phase 1 sampling) → decode.
On a lightly-trained tiny model the output is mostly gibberish; the point is that
the full tokenize→train→generate→detokenize round trip runs.

    python scripts/generate_text.py --ckpt out/ckpt_final.pt --prompt "Alice was"
"""

from __future__ import annotations

import argparse

import torch

from kimi_k3 import KimiK3Model, ModelConfig
from kimi_k3.tokenizer import ByteTokenizer


def make_tokenizer(name: str):
    if name == "byte":
        return ByteTokenizer()
    if name == "gpt2":
        from kimi_k3.tokenizer import TiktokenTokenizer

        return TiktokenTokenizer("gpt2")
    raise SystemExit(f"unknown tokenizer {name!r} (choose byte|gpt2)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="byte", choices=["byte", "gpt2"])
    ap.add_argument("--prompt", default="Alice was")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = make_tokenizer(args.tokenizer)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ModelConfig.from_dict(ckpt["model_cfg"])
    model = KimiK3Model(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ids = tok.encode(args.prompt) or [tok.bos_id]
    tokens = torch.tensor([ids], dtype=torch.long)
    gen = torch.Generator().manual_seed(args.seed)
    out = model.generate(
        tokens,
        args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_id=tok.eos_id,
        generator=gen,
    )
    text = tok.decode(out[0].tolist())
    print(text)


if __name__ == "__main__":
    main()
