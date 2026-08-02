"""Local artifact, curriculum, shard, and scheduler tests."""

import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")

from kimi_k3.pretraining.artifacts import configure_local_artifacts
from kimi_k3.pretraining.callbacks import prune_checkpoints
from kimi_k3.pretraining.config import (
    CurriculumStage,
    DataConfig,
    PretrainingConfig,
    RuntimeConfig,
    SourceConfig,
    TokenizerConfig,
)
from kimi_k3.pretraining.data import (
    PackedShardDataset,
    curriculum_source_ranges,
    validate_prepared_artifacts,
)
from kimi_k3.pretraining.trainer import (
    build_optimizer_and_scheduler,
    build_stage_trainer,
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


def test_smoke_config_is_isolated_and_small():
    production = PretrainingConfig.from_yaml("configs/pretrain_100m.yaml")
    smoke = PretrainingConfig.from_yaml("configs/pretrain_100m_smoke.yaml")
    assert smoke.artifact_root != production.artifact_root
    assert smoke.run_name != production.run_name
    assert smoke.tokenizer.artifact_name != production.tokenizer.artifact_name
    assert smoke.data.artifact_name != production.data.artifact_name
    assert smoke.data.train_tokens < production.data.train_tokens
    assert total_optimizer_steps(smoke) == 128


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


def test_muon_optimizer_path_builds_hybrid_optimizer():
    import dataclasses

    from kimi_k3.training.muon import Muon

    config = PretrainingConfig.from_yaml("configs/pretrain_100m.yaml")
    muon_config = dataclasses.replace(
        config, optimizer=dataclasses.replace(config.optimizer, name="muon")
    )
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
    optimizer, scheduler = build_optimizer_and_scheduler(model, muon_config)
    assert isinstance(optimizer, Muon)
    # Muon group (the Linear weight) and AdamW group (norm/bias) both scheduled.
    use_muon = {group["use_muon"] for group in optimizer.param_groups}
    assert use_muon == {True, False}
    assert len(scheduler.get_last_lr()) == len(optimizer.param_groups)


def test_pretraining_config_rejects_unknown_optimizer():
    import dataclasses

    config = PretrainingConfig.from_yaml("configs/pretrain_100m.yaml")
    invalid = dataclasses.replace(
        config,
        optimizer=dataclasses.replace(config.optimizer, name="unknown"),
    )
    with pytest.raises(ValueError, match="optimizer.name"):
        invalid.validate()


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepared_tiny_campaign(tmp_path: Path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    source = SourceConfig(
        name="test",
        dataset="local/test",
        subset=None,
        revision="main",
        text_field="text",
        weight=1.0,
        declared_license="test-only",
    )
    config = PretrainingConfig(
        artifact_root=str(tmp_path / "artifacts"),
        run_name="tiny-integration",
        model_preset="tiny_hybrid",
        tokenizer=TokenizerConfig(
            vocab_size=256,
            sample_bytes=1_024,
            model_max_length=32,
            artifact_name="tiny",
        ),
        runtime=RuntimeConfig(
            micro_batch_size=1,
            effective_batch_tokens=8,
            eval_steps=1,
            save_steps=1,
            logging_steps=1,
            dataloader_num_workers=0,
            bf16=False,
            keep_last_checkpoints=2,
            periodic_eval_tokens_per_source=8,
        ),
        data=DataConfig(
            train_tokens=16,
            validation_tokens=8,
            shard_tokens=16,
            artifact_name="tiny",
            sources=(source,),
        ),
        curriculum=(
            CurriculumStage(
                name="tiny-context-8",
                tokens=16,
                sequence_length=8,
                gradient_accumulation_steps=1,
            ),
        ),
    )
    config.validate()
    paths = configure_local_artifacts(config.artifact_root)
    tokenizer_dir = paths.tokenizer / config.tokenizer.artifact_name
    tokenizer_dir.mkdir(parents=True)
    vocabulary = {
        "<|pad|>": 0,
        "<|bos|>": 1,
        "<|eos|>": 2,
        "<|unk|>": 3,
        "test": 4,
    }
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<|pad|>",
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        unk_token="<|unk|>",
        model_max_length=32,
    )
    fast.save_pretrained(tokenizer_dir)
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    tokenizer_digest = _sha256(tokenizer_json)
    (tokenizer_dir / "manifest.json").write_text(
        json.dumps(
            {
                "sha256": tokenizer_digest,
                "sources": [
                    {
                        "name": "test",
                        "resolved_revision": "local-revision",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    data_dir = paths.data / config.data.artifact_name
    train_path = data_dir / "test" / "train" / "shard-00000.bin"
    validation_path = data_dir / "test" / "validation" / "shard-00000.bin"
    train_path.parent.mkdir(parents=True)
    validation_path.parent.mkdir(parents=True)
    np.asarray([4, 2] * 8, dtype="<u2").tofile(train_path)
    np.asarray([4, 2] * 4, dtype="<u2").tofile(validation_path)
    provenance_path = data_dir / "test" / "provenance.jsonl.gz"
    with gzip.open(provenance_path, "wt", encoding="utf-8") as handle:
        handle.write('{"document_sha256": "test"}\n')
    manifest = {
        "tokenizer_sha256": tokenizer_digest,
        "train_tokens": 16,
        "validation_tokens": 8,
        "sources": [
            {
                "name": "test",
                "dataset": "local/test",
                "requested_revision": "main",
                "resolved_revision": "local-revision",
                "declared_license": "test-only",
                "dataset_card_license": "test-only",
                "splits": {
                    "train": {
                        "tokens": 16,
                        "documents": 1,
                        "shards": [
                            {
                                "path": str(train_path),
                                "tokens": 16,
                                "sha256": _sha256(train_path),
                            }
                        ],
                    },
                    "validation": {
                        "tokens": 8,
                        "documents": 1,
                        "shards": [
                            {
                                "path": str(validation_path),
                                "tokens": 8,
                                "sha256": _sha256(validation_path),
                            }
                        ],
                    },
                },
                "provenance": {
                    "path": str(provenance_path),
                    "sha256": _sha256(provenance_path),
                },
            }
        ],
    }
    manifest_path = data_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return config, paths, tokenizer_dir, manifest_path, train_path


def test_prepared_artifact_validation_detects_corruption(tmp_path):
    config, _, tokenizer_dir, manifest_path, train_path = _prepared_tiny_campaign(
        tmp_path
    )
    result = validate_prepared_artifacts(
        config,
        tokenizer_dir=tokenizer_dir,
        manifest_path=manifest_path,
    )
    assert result["valid"]
    assert result["total_shards"] == 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sources"][0]["resolved_revision"] = "different-revision"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="tokenizer/data revision"):
        validate_prepared_artifacts(
            config,
            tokenizer_dir=tokenizer_dir,
            manifest_path=manifest_path,
        )
    manifest["sources"][0]["resolved_revision"] = "local-revision"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    values = np.memmap(train_path, mode="r+", dtype="<u2")
    values[0] = 3
    values.flush()
    with pytest.raises(ValueError, match="checksum"):
        validate_prepared_artifacts(
            config,
            tokenizer_dir=tokenizer_dir,
            manifest_path=manifest_path,
        )


def test_trainer_checkpoint_and_resume_roundtrip(tmp_path):
    config, paths, tokenizer_dir, manifest_path, _ = _prepared_tiny_campaign(
        tmp_path
    )
    run_dir = paths.runs / config.run_name
    first = build_stage_trainer(
        config,
        stage_index=0,
        run_dir=run_dir,
        tokenizer_dir=tokenizer_dir,
        manifest_path=manifest_path,
        max_steps_override=1,
    )
    first.train()
    checkpoint = run_dir / "checkpoint-1"
    assert first.state.global_step == 1
    assert (checkpoint / "optimizer.pt").is_file()
    assert (checkpoint / "scheduler.pt").is_file()
    assert (checkpoint / "rng_state.pth").is_file()

    resumed = build_stage_trainer(
        config,
        stage_index=0,
        run_dir=run_dir,
        tokenizer_dir=tokenizer_dir,
        manifest_path=manifest_path,
        max_steps_override=2,
    )
    resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed.state.global_step == 2
    assert resumed.lr_scheduler.last_epoch >= 2
    assert (run_dir / "checkpoint-2").is_dir()


def test_checkpoint_pruning_preserves_requested_steps(tmp_path):
    for step in range(1, 6):
        (tmp_path / f"checkpoint-{step}").mkdir()
    prune_checkpoints(tmp_path, keep_last=2, protected={2})
    assert {
        path.name for path in tmp_path.glob("checkpoint-*")
    } == {"checkpoint-2", "checkpoint-4", "checkpoint-5"}
