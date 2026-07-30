"""
kimi_k3/data/ — data pipeline for training.

Phase 2: fixed-length causal-LM packing over a token stream, with a bundled
tiny corpus for offline runs. The progressive-length curriculum (8K→64K→256K)
is future work (see docs/roadmap.md).
"""

from .text_dataset import (
    TINY_CORPUS,
    PackedTextDataset,
    batch_iterator,
    build_dataset,
    load_text,
)

__all__ = [
    "TINY_CORPUS",
    "PackedTextDataset",
    "batch_iterator",
    "build_dataset",
    "load_text",
]
