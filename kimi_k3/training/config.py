"""
kimi_k3/training/config.py — TrainConfig for the minimal trainer.

Mirrors ModelConfig's dataclass + YAML pattern. CPU/MPS-oriented defaults
(small seq_len / batch) per the Phase 2 scope; scale up on GPU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class TrainConfig:
    """Optimization + loop hyperparameters."""

    # optimizer (AdamW; Muon is a later study item)
    lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # schedule
    max_steps: int = 200
    warmup_steps: int = 20

    # data / batching
    batch_size: int = 8
    seq_len: int = 128

    # loop / io
    eval_interval: int = 25
    log_interval: int = 10
    ckpt_dir: str = "out"
    ckpt_interval: int = 100
    device: str = "auto"          # "auto" | "cpu" | "mps" | "cuda"
    seed: int = 0

    def __post_init__(self) -> None:
        positive_ints = ("max_steps", "batch_size", "seq_len")
        for name in positive_ints:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.warmup_steps < 0 or self.warmup_steps > self.max_steps:
            raise ValueError(
                f"warmup_steps must be in [0, max_steps], got {self.warmup_steps}"
            )
        if not 0.0 < self.lr:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.device not in ("auto", "cpu", "mps", "cuda"):
            raise ValueError(f"device must be auto/cpu/mps/cuda, got {self.device!r}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_yaml(self, path: str | None = None) -> str:
        import yaml

        text = yaml.safe_dump(self.to_dict(), sort_keys=True)
        if path is not None:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        return text

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        import yaml

        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)
