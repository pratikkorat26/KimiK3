"""tests/test_data.py — packing shapes and next-token label correctness."""

import pytest
import torch

from kimi_k3.data import TINY_CORPUS, PackedTextDataset, batch_iterator, build_dataset
from kimi_k3.tokenizer import ByteTokenizer


def test_packing_label_is_input_shifted():
    ids = list(range(100))
    ds = PackedTextDataset(ids, seq_len=10)
    assert len(ds) == 9  # (100 - 1) // 10
    x, y = ds[0]
    assert x.shape == (10,) and y.shape == (10,)
    # label[t] == input[t+1] over the contiguous stream
    assert torch.equal(y, x + 1)
    # second block starts where the first's inputs ended
    x1, _ = ds[1]
    assert int(x1[0]) == 10


def test_packing_requires_enough_tokens():
    with pytest.raises(ValueError, match="at least"):
        PackedTextDataset([1, 2, 3], seq_len=8)


def test_batch_iterator_shapes():
    ds = build_dataset(TINY_CORPUS, ByteTokenizer(), seq_len=16)
    x, y = next(batch_iterator(ds, batch_size=4, shuffle=False))
    assert x.shape == (4, 16) and y.shape == (4, 16)
    assert x.dtype == torch.long


def test_build_dataset_from_corpus():
    ds = build_dataset(TINY_CORPUS, ByteTokenizer(), seq_len=32)
    assert len(ds) > 1
