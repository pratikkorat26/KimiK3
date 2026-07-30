"""
kimi_k3/tokenizer/byte_tokenizer.py — zero-dependency byte-level tokenizer.

Maps each UTF-8 byte to an id in [0, 256); adds three special ids on top
(bos=256, eos=257, pad=258) for a vocab of 259. Needs no external package or
download, so the overfit test and offline machines always work. Not efficient
for real text — the tiktoken BPE is the intended path for real training.
"""

from __future__ import annotations

from .base import Tokenizer


class ByteTokenizer(Tokenizer):
    """Byte-level tokenizer: id == byte value; specials appended above 256."""

    def __init__(self) -> None:
        self.bos_id = 256
        self.eos_id = 257
        self.pad_id = 258
        self.vocab_size = 259
        self._specials = {self.bos_id, self.eos_id, self.pad_id}

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        data = bytes(i for i in ids if i not in self._specials and 0 <= i < 256)
        return data.decode("utf-8", errors="replace")
