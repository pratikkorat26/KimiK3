"""
kimi_k3/tokenizer/tiktoken_tokenizer.py — wrap an existing BPE (tiktoken).

The "reuse an existing BPE first" path: wraps a tiktoken encoding (default GPT-2,
vocab 50257) so we can train immediately instead of training our own BPE. GPT-2
has a single special token `<|endoftext|>`, used here as both BOS/EOS/PAD.

Requires `tiktoken` (install with the `train` extra: `pip install -e '.[train]'`).
"""

from __future__ import annotations

from .base import Tokenizer


class TiktokenTokenizer(Tokenizer):
    """Adapter over a tiktoken encoding."""

    def __init__(self, encoding: str = "gpt2") -> None:
        try:
            import tiktoken
        except ImportError as exc:  # pragma: no cover - exercised only without tiktoken
            raise ImportError(
                "TiktokenTokenizer requires tiktoken. Install with: pip install -e '.[train]'"
            ) from exc

        self._enc = tiktoken.get_encoding(encoding)
        self.vocab_size = self._enc.n_vocab
        self.eos_id = self._enc.eot_token          # <|endoftext|>
        self.bos_id = self.eos_id                  # GPT-2 has no separate BOS
        self.pad_id = self.eos_id

    def encode(self, text: str) -> list[int]:
        # Treat any literal special-token strings as ordinary text (never inject specials).
        return self._enc.encode(text, disallowed_special=())

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode([i for i in ids if i != self.pad_id])
