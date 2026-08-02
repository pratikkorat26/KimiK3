#!/usr/bin/env python3
"""Validate local tokenizer, token shards, provenance, and campaign totals."""

from __future__ import annotations

import argparse
import json

from kimi_k3.pretraining.artifacts import configure_local_artifacts
from kimi_k3.pretraining.config import PretrainingConfig
from kimi_k3.pretraining.data import validate_prepared_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pretrain_100m.yaml")
    parser.add_argument(
        "--skip-checksums",
        action="store_true",
        help="validate paths, sizes, and totals without hashing every file",
    )
    args = parser.parse_args()

    config = PretrainingConfig.from_yaml(args.config)
    paths = configure_local_artifacts(config.artifact_root)
    tokenizer_dir = paths.tokenizer / config.tokenizer.artifact_name
    manifest_path = paths.data / config.data.artifact_name / "manifest.json"
    result = validate_prepared_artifacts(
        config,
        tokenizer_dir=tokenizer_dir,
        manifest_path=manifest_path,
        verify_checksums=not args.skip_checksums,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
