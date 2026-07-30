"""
kimi_k3/data/text_dataset.py — pack text into causal-LM training blocks.

Pipeline: text → tokenizer.encode → one long id stream → contiguous blocks of
`seq_len`, each paired with a next-token label (label[t] == input[t+1]). Blocks
tile the stream; the trailing remainder is dropped.

A small public-domain corpus (TINY_CORPUS) is bundled so training runs offline.
Progressive-length training (8K→64K→256K) is future work — this packer is
fixed-length; a length-schedule sampler will wrap it in a later phase.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..tokenizer.base import Tokenizer

# ~1.5 KB of public-domain text (Alice's Adventures in Wonderland, ch. 1).
TINY_CORPUS = """\
Alice was beginning to get very tired of sitting by her sister on the bank, and
of having nothing to do: once or twice she had peeped into the book her sister
was reading, but it had no pictures or conversations in it, "and what is the use
of a book," thought Alice "without pictures or conversations?"

So she was considering in her own mind (as well as she could, for the hot day
made her feel very sleepy and stupid), whether the pleasure of making a
daisy-chain would be worth the trouble of getting up and picking the daisies,
when suddenly a White Rabbit with pink eyes ran close by her.

There was nothing so very remarkable in that; nor did Alice think it so very much
out of the way to hear the Rabbit say to itself, "Oh dear! Oh dear! I shall be
late!" (when she thought it over afterwards, it occurred to her that she ought to
have wondered at this, but at the time it all seemed quite natural); but when the
Rabbit actually took a watch out of its waistcoat-pocket, and looked at it, and
then hurried on, Alice started to her feet, for it flashed across her mind that
she had never before seen a rabbit with either a waistcoat-pocket, or a watch to
take out of it, and burning with curiosity, she ran across the field after it,
and fortunately was just in time to see it pop down a large rabbit-hole under the
hedge.
"""


class PackedTextDataset(Dataset):
    """Contiguous next-token blocks over a single token stream.

    Item i is (input, label) each of shape (seq_len,), int64, where
    label == the input stream shifted left by one.
    """

    def __init__(self, ids: list[int], seq_len: int):
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        if len(ids) < seq_len + 1:
            raise ValueError(
                f"need at least seq_len+1={seq_len + 1} tokens to pack, got {len(ids)}"
            )
        self.seq_len = seq_len
        self._ids = torch.tensor(ids, dtype=torch.long)
        self.n_blocks = (len(ids) - 1) // seq_len

    def __len__(self) -> int:
        return self.n_blocks

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        start = i * self.seq_len
        x = self._ids[start : start + self.seq_len]
        y = self._ids[start + 1 : start + 1 + self.seq_len]
        return x, y


def build_dataset(text: str, tokenizer: Tokenizer, seq_len: int) -> PackedTextDataset:
    """Tokenize `text` and pack it into fixed-length next-token blocks."""
    return PackedTextDataset(tokenizer.encode(text), seq_len)


def load_text(path: str) -> str:
    """Read a UTF-8 text file."""
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def batch_iterator(
    dataset: PackedTextDataset,
    batch_size: int,
    *,
    shuffle: bool = True,
    generator: torch.Generator | None = None,
) -> Iterator[tuple[Tensor, Tensor]]:
    """Yield (input, label) batches of shape (B, seq_len). Drops the last short batch.

    A plain iterator (no DataLoader workers) — simplest and portable on macOS.
    """
    n = len(dataset)
    order = (
        torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    )
    for start in range(0, n - batch_size + 1, batch_size):
        idx = order[start : start + batch_size]
        xs, ys = zip(*(dataset[int(i)] for i in idx))
        yield torch.stack(xs), torch.stack(ys)
