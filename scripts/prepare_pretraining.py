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
    tokenizer_dir = paths.tokenizer / config.tokenizer.artifact_name
    shard_dir = paths.data / config.data.artifact_name

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
        from kimi_k3.pretraining.data import (
            prepare_token_shards,
            validate_prepared_artifacts,
        )

        manifest = prepare_token_shards(
            config,
            tokenizer_dir=tokenizer_dir,
            output_dir=shard_dir,
        )
        validation = validate_prepared_artifacts(
            config,
            tokenizer_dir=tokenizer_dir,
            manifest_path=shard_dir / "manifest.json",
        )
        print(
            f"shards: {shard_dir} "
            f"({manifest['train_tokens']:,} train tokens, "
            f"{validation['total_shards']} verified shards)"
        )


if __name__ == "__main__":
    main()
