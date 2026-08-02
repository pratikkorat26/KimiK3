"""YAML configuration for the local 100M pretraining campaign."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SourceConfig:
    name: str
    dataset: str
    subset: str | None
    revision: str
    text_field: str
    weight: float
    declared_license: str


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    tokens: int
    sequence_length: int
    gradient_accumulation_steps: int


@dataclass(frozen=True)
class TokenizerConfig:
    vocab_size: int = 32_000
    sample_bytes: int = 100_000_000
    model_max_length: int = 4096
    artifact_name: str = "v1"


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "adamw"                 # "adamw" or "muon" (Per-Head Muon + AdamW)
    learning_rate: float = 3e-4
    final_learning_rate: float = 3e-5
    warmup_ratio: float = 0.01
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0
    muon_learning_rate: float = 0.02    # Muon-only (ignored when name == "adamw")
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 42
    micro_batch_size: int = 1
    effective_batch_tokens: int = 32_768
    eval_steps: int = 250
    save_steps: int = 250
    logging_steps: int = 10
    dataloader_num_workers: int = 2
    bf16: bool = True
    max_elapsed_days: float = 30.0
    keep_last_checkpoints: int = 3
    periodic_eval_tokens_per_source: int = 131_072


@dataclass(frozen=True)
class DataConfig:
    train_tokens: int = 500_000_000
    validation_tokens: int = 5_000_000
    shard_tokens: int = 10_000_000
    artifact_name: str = "tokenized-v1"
    sources: tuple[SourceConfig, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PretrainingConfig:
    artifact_root: str = ".artifacts"
    run_name: str = "kimi-k3-100m-pretrain"
    model_preset: str = "small"
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    curriculum: tuple[CurriculumStage, ...] = field(default_factory=tuple)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PretrainingConfig:
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise ValueError("pretraining config must be a YAML mapping")

        tokenizer = TokenizerConfig(**raw.get("tokenizer", {}))
        optimizer = OptimizerConfig(**raw.get("optimizer", {}))
        runtime = RuntimeConfig(**raw.get("runtime", {}))
        data_raw = raw.get("data", {})
        sources = tuple(SourceConfig(**item) for item in data_raw.pop("sources", []))
        data = DataConfig(sources=sources, **data_raw)
        curriculum = tuple(
            CurriculumStage(**item) for item in raw.get("curriculum", [])
        )
        config = cls(
            artifact_root=raw.get("artifact_root", ".artifacts"),
            run_name=raw.get("run_name", "kimi-k3-100m-pretrain"),
            model_preset=raw.get("model_preset", "small"),
            tokenizer=tokenizer,
            optimizer=optimizer,
            runtime=runtime,
            data=data,
            curriculum=curriculum,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.model_preset not in {"small", "tiny_hybrid"}:
            raise ValueError(
                "local pretraining supports model_preset='small' or "
                "'tiny_hybrid'"
            )
        if self.optimizer.name not in {"adamw", "muon"}:
            raise ValueError("optimizer.name must be 'adamw' or 'muon'")
        for optimizer_label, optimizer_value in (
            ("optimizer.learning_rate", self.optimizer.learning_rate),
            ("optimizer.final_learning_rate", self.optimizer.final_learning_rate),
            ("optimizer.muon_learning_rate", self.optimizer.muon_learning_rate),
            ("optimizer.max_grad_norm", self.optimizer.max_grad_norm),
        ):
            if optimizer_value <= 0:
                raise ValueError(f"{optimizer_label} must be positive")
        if not 0.0 <= self.optimizer.warmup_ratio <= 1.0:
            raise ValueError("optimizer.warmup_ratio must be in [0, 1]")
        if self.optimizer.weight_decay < 0:
            raise ValueError("optimizer.weight_decay must be non-negative")
        if not 0.0 <= self.optimizer.muon_momentum < 1.0:
            raise ValueError("optimizer.muon_momentum must be in [0, 1)")
        if self.optimizer.muon_ns_steps <= 0:
            raise ValueError("optimizer.muon_ns_steps must be positive")
        if self.tokenizer.vocab_size < 256:
            raise ValueError("tokenizer vocab_size must be at least 256")
        if self.tokenizer.sample_bytes <= 0:
            raise ValueError("tokenizer.sample_bytes must be positive")
        if self.tokenizer.model_max_length <= 0:
            raise ValueError("tokenizer.model_max_length must be positive")
        for artifact_label, artifact_name in (
            ("tokenizer.artifact_name", self.tokenizer.artifact_name),
            ("data.artifact_name", self.data.artifact_name),
        ):
            if (
                not artifact_name
                or Path(artifact_name).name != artifact_name
                or artifact_name in {".", ".."}
            ):
                raise ValueError(
                    f"{artifact_label} must be a single directory name"
                )
        if not self.data.sources:
            raise ValueError("at least one data source is required")
        if len({source.name for source in self.data.sources}) != len(
            self.data.sources
        ):
            raise ValueError("source names must be unique")
        if any(source.weight <= 0 for source in self.data.sources):
            raise ValueError("source weights must be positive")
        for numeric_label, numeric_value in (
            ("data.train_tokens", self.data.train_tokens),
            ("data.validation_tokens", self.data.validation_tokens),
            ("data.shard_tokens", self.data.shard_tokens),
            ("runtime.micro_batch_size", self.runtime.micro_batch_size),
            (
                "runtime.effective_batch_tokens",
                self.runtime.effective_batch_tokens,
            ),
        ):
            if numeric_value <= 0:
                raise ValueError(f"{numeric_label} must be positive")
        weight = sum(source.weight for source in self.data.sources)
        if abs(weight - 1.0) > 1e-9:
            raise ValueError(f"source weights must sum to 1.0, got {weight}")
        if sum(stage.tokens for stage in self.curriculum) != self.data.train_tokens:
            raise ValueError("curriculum tokens must equal data.train_tokens")
        for stage in self.curriculum:
            if (
                stage.tokens <= 0
                or stage.sequence_length <= 0
                or stage.gradient_accumulation_steps <= 0
            ):
                raise ValueError(
                    f"curriculum stage {stage.name!r} values must be positive"
                )
            effective = (
                self.runtime.micro_batch_size
                * stage.sequence_length
                * stage.gradient_accumulation_steps
            )
            if effective != self.runtime.effective_batch_tokens:
                raise ValueError(
                    f"stage {stage.name!r} effective batch must be "
                    f"{self.runtime.effective_batch_tokens} tokens, got {effective}"
                )

    def stage_source_tokens(self, stage_index: int, source_index: int) -> int:
        stage = self.curriculum[stage_index]
        source = self.data.sources[source_index]
        if source_index == len(self.data.sources) - 1:
            prior = sum(
                int(stage.tokens * item.weight)
                for item in self.data.sources[:-1]
            )
            return stage.tokens - prior
        return int(stage.tokens * source.weight)

    def source_train_tokens(self, source_index: int) -> int:
        return sum(
            self.stage_source_tokens(stage_index, source_index)
            for stage_index in range(len(self.curriculum))
        )

    def source_validation_tokens(self, source_index: int) -> int:
        if source_index == len(self.data.sources) - 1:
            prior = sum(
                int(self.data.validation_tokens * item.weight)
                for item in self.data.sources[:-1]
            )
            return self.data.validation_tokens - prior
        return int(self.data.validation_tokens * self.data.sources[source_index].weight)

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_root": self.artifact_root,
            "run_name": self.run_name,
            "model_preset": self.model_preset,
        }
