"""
kimi_k3/tokenizer/ — text tokenizers for training.

Phase 2 reuses an existing BPE rather than training one from scratch:
  - ByteTokenizer     — zero-dependency byte-level (vocab 259); always available.
  - TiktokenTokenizer — wraps a tiktoken BPE (default GPT-2, vocab 50257).

Both implement the `Tokenizer` protocol (encode / decode / vocab_size / bos/eos/pad).
Training our own BPE remains future work (see docs/roadmap.md).
"""

from .base import Tokenizer
from .byte_tokenizer import ByteTokenizer

__all__ = ["ByteTokenizer", "Tokenizer", "TiktokenTokenizer"]


def __getattr__(name: str):
    # Lazy import so the package works without tiktoken installed.
    if name == "TiktokenTokenizer":
        from .tiktoken_tokenizer import TiktokenTokenizer

        return TiktokenTokenizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
