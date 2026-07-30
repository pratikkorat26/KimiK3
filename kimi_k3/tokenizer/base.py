"""
kimi_k3/tokenizer/base.py — the Tokenizer protocol.

Every tokenizer used by the trainer implements this minimal surface. The model's
vocab_size is set from the tokenizer's `vocab_size`, so embeddings and lm_head match.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """Minimal text <-> token-id interface."""

    vocab_size: int
    bos_id: int
    eos_id: int
    pad_id: int

    def encode(self, text: str) -> list[int]:
        """Text → token ids (no special tokens added)."""
        ...

    def decode(self, ids: list[int]) -> str:
        """Token ids → text (special tokens dropped)."""
        ...
