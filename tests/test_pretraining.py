"""Local artifact, curriculum, shard, and scheduler tests."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")

from kimi_k3.pretraining.artifacts import configure_local_artifacts
from kimi_k3.pretraining.config import PretrainingConfig
from kimi_k3.pretraining.data import PackedShardDataset, curriculum_source_ranges
from kimi_k3.pretraining.trainer import (
    build_optimizer_and_scheduler,
    build_training_arguments,
    stage_end_step,
    total_optimizer_steps,
)


def test_pretraining_config_and_curriculum_accounting():
    config = PretrainingConfig.from_yaml("configs/pretrain_100m.yaml")
    assert config.data.train_tokens == 500_006_912
    assert total_optimizer_steps(config) == 15_259
    assert stage_end_step(config, 0) == 10_681
    assert stage_end_step(config, 1) == 13_733
    assert stage_end_step(config, 2) == 15_259
    assert sum(
        config.source_train_tokens(index)
        for index in range(len(config.data.sources))
    ) == config.data.train_tokens

    first = curriculum_source_ranges(config, 0)
    second = curriculum_source_ranges(config, 1)
    for source in config.data.sources:
        assert second[source.name][0] == sum(first[source.name])


def test_artifact_environment_is_project_local(tmp_path, monkeypatch):
    for name in ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TMPDIR", "TMP", "TEMP"):
        monkeypatch.delenv(name, raising=False)
    paths = configure_local_artifacts(tmp_path / "artifacts")
    assert Path(os.environ["HF_HOME"]).is_relative_to(paths.root)
    assert Path(os.environ["HF_DATASETS_CACHE"]).is_relative_to(paths.root)
    assert Path(os.environ["TMPDIR"]).is_relative_to(paths.root)


def test_packed_shard_dataset_honors_source_ranges(tmp_path):
    shard_a = tmp_path / "a.bin"
    shard_b = tmp_path / "b.bin"
    np.arange(32, dtype="<u2").tofile(shard_a)
    np.arange(100, 132, dtype="<u2").tofile(shard_b)
    manifest = {
        "sources": [
            {
                "name": "english",
                "splits": {
                    "train": {
                        "tokens": 32,
                        "shards": [{"path": str(shard_a), "tokens": 32}],
                    }
                },
            },
            {
                "name": "code",
                "splits": {
                    "train": {
                        "tokens": 32,
                        "shards": [{"path": str(shard_b), "tokens": 32}],
                    }
                },
            },
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    dataset = PackedShardDataset(
        manifest_path,
        split="train",
        sequence_length=4,
        source_ranges={"english": (8, 8), "code": (4, 8)},
    )
    assert len(dataset) == 4
    assert torch.equal(dataset[0]["input_ids"], torch.tensor([8, 9, 10, 11]))
    assert torch.equal(dataset[2]["input_ids"], torch.tensor([104, 105, 106, 107]))
    assert torch.equal(dataset[0]["input_ids"], dataset[0]["labels"])


def test_scheduler_warms_up_and_reaches_floor():
    config = PretrainingConfig.from_yaml("configs/pretrain_100m.yaml")
    model = torch.nn.Linear(2, 2)
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)
    initial_lr = scheduler.get_last_lr()[0]
    assert 0 < initial_lr < config.optimizer.learning_rate

    for _ in range(total_optimizer_steps(config)):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(
        config.optimizer.final_learning_rate,
        rel=1e-3,
    )


def test_training_arguments_enable_monitoring_and_checkpointing(tmp_path):
    config = PretrainingConfig.from_yaml("configs/pretrain_100m.yaml")
    arguments = build_training_arguments(
        config,
        stage_index=0,
        run_dir=tmp_path,
        end_step=10,
    )
    assert arguments.bf16
    assert arguments.gradient_checkpointing
    assert arguments.eval_steps == 250
    assert arguments.save_steps == 250
    assert arguments.logging_dir == str(tmp_path / "tensorboard")
