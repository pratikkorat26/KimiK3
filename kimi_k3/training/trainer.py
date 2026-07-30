"""
kimi_k3/training/trainer.py — minimal AdamW trainer for KimiK3Model.

Scope (Phase 2): correctness over throughput. Runs on CPU (default) or an
explicitly-requested MPS/CUDA device. Training uses the parallel chunk path
(mode="chunk"); the Quantile-Balancing router bias update fires automatically
under model.train(). Optimizer is AdamW — Per-Head Muon (K3's optimizer) is a
later study item.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Iterator

import torch
from torch import Tensor

from ..config import ModelConfig
from ..data.text_dataset import PackedTextDataset, batch_iterator
from ..model import KimiK3Model
from .config import TrainConfig
from .loss import causal_lm_loss, perplexity


def resolve_device(name: str) -> torch.device:
    """Map a TrainConfig.device string to a concrete torch.device.

    "auto" prefers cuda, then Apple MPS, then cpu. The KDA triangular-solve has an
    MPS fallback (see _unit_lower_tri_inverse), so MPS is a safe default on a Mac;
    force cpu with device="cpu" if any op is unsupported on your torch build.
    """
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device='mps' requested but MPS is not available")
    return torch.device(name)


class Trainer:
    """AdamW trainer with warmup+cosine LR, grad clipping, and checkpointing."""

    def __init__(self, model: KimiK3Model, cfg: TrainConfig):
        self.cfg = cfg
        self.model_cfg: ModelConfig = model.cfg
        self.device = resolve_device(cfg.device)
        self.model = model.to(self.device)
        self.step = 0

        # Weight decay on matrices only (not norms/biases/1-D params).
        decay: list[torch.nn.Parameter] = []
        no_decay: list[torch.nn.Parameter] = []
        for p in model.parameters():
            if not p.requires_grad:
                continue
            (decay if p.ndim >= 2 else no_decay).append(p)
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": cfg.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
        )

    # --- LR schedule ----------------------------------------------------
    def _lr_at(self, step: int) -> float:
        cfg = self.cfg
        if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
            return cfg.lr * (step + 1) / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return cfg.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))

    def _apply_lr(self, step: int) -> float:
        lr = self._lr_at(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    # --- one optimization step -----------------------------------------
    def train_step(self, x: Tensor, y: Tensor) -> float:
        self.model.train()
        x, y = x.to(self.device), y.to(self.device)
        self._apply_lr(self.step)
        logits, _ = self.model(x, mode="chunk")
        loss = causal_lm_loss(logits, y)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        self.step += 1
        return float(loss.detach())

    @torch.no_grad()
    def evaluate(self, batches: Iterator[tuple[Tensor, Tensor]], max_batches: int = 10) -> tuple[float, float]:
        self.model.eval()
        total, n = 0.0, 0
        for i, (x, y) in enumerate(batches):
            if i >= max_batches:
                break
            x, y = x.to(self.device), y.to(self.device)
            logits, _ = self.model(x, mode="chunk")
            total += float(causal_lm_loss(logits, y))
            n += 1
        avg = total / max(1, n)
        return avg, perplexity(torch.tensor(avg))

    # --- full training loop over a dataset -----------------------------
    def fit(
        self,
        train_ds: PackedTextDataset,
        val_ds: PackedTextDataset | None = None,
    ) -> None:
        """Train for cfg.max_steps, cycling batches from train_ds."""
        if len(train_ds) < self.cfg.batch_size:
            raise ValueError(
                f"train dataset has {len(train_ds)} blocks < batch_size "
                f"{self.cfg.batch_size}; use a longer corpus or smaller seq_len/batch"
            )
        gen = torch.Generator().manual_seed(self.cfg.seed)
        batches = self._cycle(train_ds, gen)
        window_start = time.time()
        while self.step < self.cfg.max_steps:
            x, y = next(batches)
            loss = self.train_step(x, y)
            step = self.step
            if step % self.cfg.log_interval == 0:
                dt = time.time() - window_start
                toks = self.cfg.batch_size * self.cfg.seq_len * self.cfg.log_interval
                tps = toks / dt if dt > 0 else 0.0
                print(
                    f"step {step:5d} | loss {loss:.4f} | lr {self._lr_at(step):.2e} "
                    f"| {tps:,.0f} tok/s"
                )
                window_start = time.time()
            if val_ds is not None and step % self.cfg.eval_interval == 0:
                vloss, vppl = self.evaluate(batch_iterator(val_ds, self.cfg.batch_size, shuffle=False))
                print(f"  eval @ {step}: loss {vloss:.4f} | ppl {vppl:.1f}")
            if step % self.cfg.ckpt_interval == 0:
                self.save_checkpoint()
        self.save_checkpoint(tag="final")

    def _cycle(
        self, dataset: PackedTextDataset, generator: torch.Generator
    ) -> Iterator[tuple[Tensor, Tensor]]:
        """Infinite reshuffled batch stream."""
        while True:
            yield from batch_iterator(dataset, self.cfg.batch_size, generator=generator)

    # --- overfit helper (the key sanity gate) --------------------------
    def overfit(self, x: Tensor, y: Tensor, steps: int) -> list[float]:
        """Repeatedly train on one fixed batch; return the loss history."""
        return [self.train_step(x, y) for _ in range(steps)]

    # --- checkpointing --------------------------------------------------
    def save_checkpoint(self, tag: str | None = None) -> str:
        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)
        name = f"ckpt_{tag or self.step}.pt"
        path = os.path.join(self.cfg.ckpt_dir, name)
        torch.save(
            {
                "step": self.step,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "model_cfg": self.model_cfg.to_dict(),
                "train_cfg": self.cfg.to_dict(),
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.step = ckpt["step"]
