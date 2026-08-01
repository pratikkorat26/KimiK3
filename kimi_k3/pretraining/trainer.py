"""Hugging Face Trainer construction for the staged local campaign."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers import Trainer, TrainingArguments

from ..config import ModelConfig
from ..hf import KimiK3ForCausalLM, KimiK3HFConfig
from ..training.muon import build_muon_optimizer
from .callbacks import (
    ElapsedTimeLimitCallback,
    LocalMetricsCallback,
    StageBoundaryCallback,
)
from .config import PretrainingConfig
from .data import PackedShardDataset, curriculum_source_ranges
from .tokenizer import load_tokenizer


def total_optimizer_steps(config: PretrainingConfig) -> int:
    return sum(
        math.ceil(
            stage.tokens
            / (
                config.runtime.micro_batch_size
                * stage.sequence_length
                * stage.gradient_accumulation_steps
            )
        )
        for stage in config.curriculum
    )


def stage_end_step(config: PretrainingConfig, stage_index: int) -> int:
    return sum(
        math.ceil(
            stage.tokens
            / (
                config.runtime.micro_batch_size
                * stage.sequence_length
                * stage.gradient_accumulation_steps
            )
        )
        for stage in config.curriculum[: stage_index + 1]
    )


def build_model(config: PretrainingConfig, tokenizer) -> KimiK3ForCausalLM:
    if config.model_preset != "small":
        raise ValueError(f"unsupported model preset {config.model_preset!r}")
    core = ModelConfig.small()
    core.vocab_size = len(tokenizer)
    hf_config = KimiK3HFConfig.from_model_config(
        core,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return KimiK3ForCausalLM(hf_config)


def build_optimizer_and_scheduler(
    model: torch.nn.Module,
    config: PretrainingConfig,
) -> tuple[torch.optim.Optimizer, LambdaLR]:
    if config.optimizer.name == "muon":
        # Per-Head Muon on hidden matmuls + AdamW on embeddings/head/norms.
        optimizer: torch.optim.Optimizer = build_muon_optimizer(
            model,
            muon_lr=config.optimizer.muon_learning_rate,
            adam_lr=config.optimizer.learning_rate,
            momentum=config.optimizer.muon_momentum,
            ns_steps=config.optimizer.muon_ns_steps,
            betas=(config.optimizer.beta1, config.optimizer.beta2),
            weight_decay=config.optimizer.weight_decay,
        )
    else:
        decay: list[torch.nn.Parameter] = []
        no_decay: list[torch.nn.Parameter] = []
        for parameter in model.parameters():
            if not parameter.requires_grad:
                continue
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": config.optimizer.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=config.optimizer.learning_rate,
            betas=(config.optimizer.beta1, config.optimizer.beta2),
            fused=torch.cuda.is_available(),
        )
    total_steps = total_optimizer_steps(config)
    warmup_steps = max(1, int(total_steps * config.optimizer.warmup_ratio))
    floor = config.optimizer.final_learning_rate / config.optimizer.learning_rate

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(
            max((step - warmup_steps) / max(1, total_steps - warmup_steps), 0.0),
            1.0,
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor + (1.0 - floor) * cosine

    return optimizer, LambdaLR(optimizer, multiplier)


def build_training_arguments(
    config: PretrainingConfig,
    *,
    stage_index: int,
    run_dir: Path,
    end_step: int,
) -> TrainingArguments:
    stage = config.curriculum[stage_index]
    return TrainingArguments(
        output_dir=str(run_dir),
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=config.runtime.eval_steps,
        save_strategy="steps",
        save_steps=config.runtime.save_steps,
        save_total_limit=config.runtime.keep_last_checkpoints,
        logging_strategy="steps",
        logging_steps=config.runtime.logging_steps,
        logging_first_step=True,
        report_to=["tensorboard"],
        logging_dir=str(run_dir / "tensorboard"),
        per_device_train_batch_size=config.runtime.micro_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=stage.gradient_accumulation_steps,
        max_steps=end_step,
        learning_rate=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        max_grad_norm=config.optimizer.max_grad_norm,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        dataloader_num_workers=config.runtime.dataloader_num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        seed=config.runtime.seed,
        data_seed=config.runtime.seed,
        ignore_data_skip=True,
        prediction_loss_only=True,
    )


def build_stage_trainer(
    config: PretrainingConfig,
    *,
    stage_index: int,
    run_dir: Path,
    tokenizer_dir: Path,
    manifest_path: Path,
    max_steps_override: int | None = None,
) -> Trainer:
    stage = config.curriculum[stage_index]
    tokenizer = load_tokenizer(tokenizer_dir)
    model = build_model(config, tokenizer)
    model.gradient_checkpointing_enable()

    train_dataset = PackedShardDataset(
        manifest_path,
        split="train",
        sequence_length=stage.sequence_length,
        source_ranges=curriculum_source_ranges(config, stage_index),
    )
    eval_datasets = {
        source.name: PackedShardDataset(
            manifest_path,
            split="validation",
            sequence_length=stage.sequence_length,
            source_ranges={
                source.name: (
                    0,
                    min(
                        config.runtime.periodic_eval_tokens_per_source,
                        config.source_validation_tokens(source_index),
                    ),
                )
            },
            source_name=source.name,
        )
        for source_index, source in enumerate(config.data.sources)
    }
    end_step = max_steps_override or stage_end_step(config, stage_index)
    args = build_training_arguments(
        config,
        stage_index=stage_index,
        run_dir=run_dir,
        end_step=end_step,
    )
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)
    tokens_per_step = (
        config.runtime.micro_batch_size
        * stage.sequence_length
        * stage.gradient_accumulation_steps
    )
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        processing_class=tokenizer,
        optimizers=(optimizer, scheduler),
        callbacks=[
            LocalMetricsCallback(
                path=run_dir / "metrics.jsonl",
                model=model,
                tokens_per_step=tokens_per_step,
            ),
            ElapsedTimeLimitCallback(
                run_dir,
                config.runtime.max_elapsed_days,
            ),
            StageBoundaryCallback(end_step),
        ],
    )
