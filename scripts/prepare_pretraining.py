#!/usr/bin/env python3
"""Prepare the tokenizer and deterministic local token shards."""

from __future__ import annotations

import argparse

from kimi_k3.pretraining.artifacts import configure_local_artifacts
from kimi_k3.pretraining.config import PretrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pretrain_100m.yaml")
    parser.add_argument(
        "--step",
        choices=("tokenizer", "shards", "all"),
        default="all",
    )
    parser.add_argument(
        "--confirm-download",
        action="store_true",
        help="required before streaming source datasets",
    )
    args = parser.parse_args()
    config = PretrainingConfig.from_yaml(args.config)
    paths = configure_local_artifacts(config.artifact_root)
    tokenizer_dir = paths.tokenizer / "v1"
    shard_dir = paths.data / "tokenized-v1"

    if not args.confirm_download:
        raise SystemExit(
            "data preparation streams remote datasets; rerun with --confirm-download"
        )
    if args.step in ("tokenizer", "all"):
        from kimi_k3.pretraining.tokenizer import train_tokenizer

        manifest = train_tokenizer(config, tokenizer_dir)
        print(f"tokenizer: {tokenizer_dir} ({manifest['vocab_size']} tokens)")
    if args.step in ("shards", "all"):
        if not (tokenizer_dir / "tokenizer.json").exists():
            raise SystemExit(f"tokenizer is missing: {tokenizer_dir}")
        from kimi_k3.pretraining.data import prepare_token_shards

        manifest = prepare_token_shards(
            config,
            tokenizer_dir=tokenizer_dir,
            output_dir=shard_dir,
        )
        print(
            f"shards: {shard_dir} "
            f"({manifest['train_tokens']:,} train tokens)"
        )


if __name__ == "__main__":
    main()
