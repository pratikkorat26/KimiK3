"""Trainer callbacks for local metrics, time limits, and stage checkpoints."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback

from ..moe.router import QuantileBalancingRouter


class LocalMetricsCallback(TrainerCallback):
    def __init__(
        self,
        *,
        path: Path,
        model: torch.nn.Module,
        tokens_per_step: int,
    ) -> None:
        self.path = path
        self.model = model
        self.tokens_per_step = tokens_per_step
        self.last_time = time.monotonic()
        self.last_step = 0
        path.parent.mkdir(parents=True, exist_ok=True)

    def on_train_begin(self, args, state, control, **kwargs):
        del args, control, kwargs
        self.last_step = state.global_step
        self.last_time = time.monotonic()

    def on_log(self, args, state, control, logs=None, **kwargs):
        del args, control, kwargs
        if logs is None or not state.is_world_process_zero:
            return
        now = time.monotonic()
        step_delta = state.global_step - self.last_step
        elapsed = max(now - self.last_time, 1e-9)
        record: dict[str, Any] = dict(logs)
        record["global_step"] = state.global_step
        record["tokens_per_second_window"] = (
            step_delta * self.tokens_per_step / elapsed
        )
        record["tokens_seen"] = state.global_step * self.tokens_per_step
        if torch.cuda.is_available():
            record["gpu_memory_allocated_gib"] = (
                torch.cuda.memory_allocated() / 1024**3
            )
            record["gpu_memory_reserved_gib"] = (
                torch.cuda.memory_reserved() / 1024**3
            )

        counts = [
            module.last_expert_counts.detach().float().cpu()
            for module in self.model.modules()
            if isinstance(module, QuantileBalancingRouter)
        ]
        if counts:
            combined = torch.stack(counts).sum(dim=0)
            mean = float(combined.mean())
            record["router_min_to_mean"] = (
                float(combined.min()) / mean if mean else 0.0
            )
            record["router_max_to_mean"] = (
                float(combined.max()) / mean if mean else 0.0
            )

        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.last_time = now
        self.last_step = state.global_step


class ElapsedTimeLimitCallback(TrainerCallback):
    def __init__(self, run_dir: Path, max_days: float) -> None:
        self.path = run_dir / "run-start.json"
        self.max_seconds = max_days * 24 * 60 * 60
        if self.path.exists():
            value = json.loads(self.path.read_text(encoding="utf-8"))
            self.started_at = float(value["unix_time"])
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.started_at = time.time()
            self.path.write_text(
                json.dumps({"unix_time": self.started_at}, indent=2) + "\n",
                encoding="utf-8",
            )

    def on_step_end(self, args, state, control, **kwargs):
        del args, state, kwargs
        if time.time() - self.started_at >= self.max_seconds:
            control.should_save = True
            control.should_training_stop = True
        return control


class StageBoundaryCallback(TrainerCallback):
    def __init__(self, end_step: int) -> None:
        self.end_step = end_step

    def on_step_end(self, args, state, control, **kwargs):
        del args, kwargs
        if state.global_step >= self.end_step:
            control.should_save = True
            control.should_training_stop = True
        return control


def prune_checkpoints(run_dir: Path, keep_last: int, protected: set[int]) -> None:
    checkpoints: list[tuple[int, Path]] = []
    for path in run_dir.glob("checkpoint-*"):
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except ValueError:
            continue
        checkpoints.append((step, path))
    checkpoints.sort()
    keep = {step for step, _ in checkpoints[-keep_last:]} | protected
    for step, path in checkpoints:
        if step not in keep:
            shutil.rmtree(path)
