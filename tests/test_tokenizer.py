"""tests/test_tokenizer.py — tokenizer roundtrips and the Tokenizer protocol."""

import pytest

from kimi_k3.tokenizer import ByteTokenizer, Tokenizer


def test_byte_tokenizer_roundtrip():
    tok = ByteTokenizer()
    text = "Hello, world! — café 🚀"
    assert tok.decode(tok.encode(text)) == text


def test_byte_tokenizer_vocab_and_specials():
    tok = ByteTokenizer()
    assert tok.vocab_size == 259
    assert {tok.bos_id, tok.eos_id, tok.pad_id} == {256, 257, 258}
    # encode never emits special ids
    assert all(0 <= i < 256 for i in tok.encode("abc"))
    # decode drops specials
    assert tok.decode([ord("h"), ord("i"), tok.eos_id]) == "hi"


def test_byte_tokenizer_satisfies_protocol():
    assert isinstance(ByteTokenizer(), Tokenizer)


def test_tiktoken_optional():
    tiktoken = pytest.importorskip("tiktoken")
    from kimi_k3.tokenizer import TiktokenTokenizer

    tok = TiktokenTokenizer("gpt2")
    assert tok.vocab_size == tok._enc.n_vocab
    assert tok.decode(tok.encode("hello world")) == "hello world"
    del tiktoken
