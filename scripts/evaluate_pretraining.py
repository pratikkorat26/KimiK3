#!/usr/bin/env python3
"""Evaluate a local pretraining checkpoint by domain and generate samples."""

from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from kimi_k3.hf import KimiK3ForCausalLM
from kimi_k3.pretraining.artifacts import configure_local_artifacts
from kimi_k3.pretraining.config import PretrainingConfig
from kimi_k3.pretraining.data import (
    PackedShardDataset,
    validate_prepared_artifacts,
)
from kimi_k3.pretraining.tokenizer import load_tokenizer


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    return device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pretrain_100m.yaml")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--tokens-per-source", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sample-new-tokens", type=int, default=32)
    args = parser.parse_args()

    config = PretrainingConfig.from_yaml(args.config)
    if args.stage < 0 or args.stage >= len(config.curriculum):
        raise SystemExit(f"--stage must be in [0, {len(config.curriculum) - 1}]")
    if args.tokens_per_source is not None and args.tokens_per_source <= 0:
        raise SystemExit("--tokens-per-source must be positive")
    if args.sample_new_tokens < 0:
        raise SystemExit("--sample-new-tokens must be non-negative")

    paths = configure_local_artifacts(config.artifact_root)
    tokenizer_dir = paths.tokenizer / config.tokenizer.artifact_name
    manifest_path = paths.data / config.data.artifact_name / "manifest.json"
    validate_prepared_artifacts(
        config,
        tokenizer_dir=tokenizer_dir,
        manifest_path=manifest_path,
        verify_checksums=False,
    )

    device = _device(args.device)
    tokenizer = load_tokenizer(tokenizer_dir)
    model = KimiK3ForCausalLM.from_pretrained(Path(args.model_path)).eval()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model.to(device=device, dtype=dtype)

    stage = config.curriculum[args.stage]
    token_limit = (
        args.tokens_per_source
        or config.runtime.periodic_eval_tokens_per_source
    )
    results: dict[str, object] = {
        "model_path": str(Path(args.model_path).resolve()),
        "device": str(device),
        "stage": stage.name,
        "sequence_length": stage.sequence_length,
        "domains": {},
    }
    domains: dict[str, object] = {}
    for source_index, source in enumerate(config.data.sources):
        available = config.source_validation_tokens(source_index)
        count = min(token_limit, available)
        dataset = PackedShardDataset(
            manifest_path,
            split="validation",
            sequence_length=stage.sequence_length,
            source_ranges={source.name: (0, count)},
            source_name=source.name,
        )
        sequences = min(len(dataset), max(1, count // stage.sequence_length))
        loader = DataLoader(
            Subset(dataset, range(sequences)),
            batch_size=1,
            shuffle=False,
        )
        loss_sum = 0.0
        tokens = 0
        precision_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), precision_context:
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                output = model(input_ids=input_ids, labels=input_ids)
                if output.loss is None or not bool(torch.isfinite(output.loss)):
                    raise RuntimeError(
                        f"non-finite validation loss for source {source.name!r}"
                    )
                predicted = input_ids.numel() - input_ids.shape[0]
                loss_sum += float(output.loss) * predicted
                tokens += predicted
        loss = loss_sum / tokens
        domains[source.name] = {
            "loss": loss,
            "perplexity": math.exp(min(loss, 50.0)),
            "predicted_tokens": tokens,
        }
    results["domains"] = domains

    prompts = (
        "The key idea is",
        "def binary_search",
        "To solve the equation",
    )
    samples = []
    for prompt in prompts:
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(device)
        generated = model.model.generate(
            encoded,
            args.sample_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )
        samples.append(
            {
                "prompt": prompt,
                "text": tokenizer.decode(
                    generated[0],
                    skip_special_tokens=True,
                ),
            }
        )
    results["samples"] = samples
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
