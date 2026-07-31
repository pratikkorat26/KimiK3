"""Project-local paths and environment variables for pretraining artifacts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path

    @classmethod
    def from_root(cls, root: str | Path) -> ArtifactPaths:
        return cls(Path(root).expanduser().resolve())

    @property
    def hf(self) -> Path:
        return self.root / "hf"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def tokenizer(self) -> Path:
        return self.root / "tokenizer"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def tmp(self) -> Path:
        return self.root / "tmp"

    def create(self) -> None:
        for path in (
            self.root,
            self.hf,
            self.hf / "hub",
            self.hf / "datasets",
            self.data,
            self.tokenizer,
            self.runs,
            self.tmp,
        ):
            path.mkdir(parents=True, exist_ok=True)


def configure_local_artifacts(root: str | Path) -> ArtifactPaths:
    """Route Hugging Face caches and temporary files beneath ``root``."""
    paths = ArtifactPaths.from_root(root)
    paths.create()
    values = {
        "HF_HOME": paths.hf,
        "HF_HUB_CACHE": paths.hf / "hub",
        "HF_DATASETS_CACHE": paths.hf / "datasets",
        "TMPDIR": paths.tmp,
        "TMP": paths.tmp,
        "TEMP": paths.tmp,
    }
    for name, path in values.items():
        os.environ[name] = str(path)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    return paths
