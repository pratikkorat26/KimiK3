"""Train and load the fixed 32K byte-level BPE tokenizer."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from .config import PretrainingConfig, SourceConfig

PAD_TOKEN = "<|pad|>"
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"
UNK_TOKEN = "<|unk|>"
ROLE_TOKENS = (
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool_call|>",
    "<|tool_result|>",
)
SPECIAL_TOKENS = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, *ROLE_TOKENS)


def clean_document(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(text) < 64:
        return None
    return text


def _stream_source(source: SourceConfig, revision: str):
    from datasets import load_dataset

    dataset = load_dataset(
        source.dataset,
        source.subset,
        split="train",
        revision=revision,
        streaming=True,
    )
    if source.text_field not in dataset.column_names:
        raise ValueError(
            f"{source.dataset} does not contain text field {source.text_field!r}; "
            f"available fields: {dataset.column_names}"
        )
    return dataset


def _weighted_text_iterator(
    config: PretrainingConfig,
    revisions: dict[str, str],
) -> Iterator[str]:
    remaining = [
        int(config.tokenizer.sample_bytes * source.weight)
        for source in config.data.sources
    ]
    remaining[-1] += config.tokenizer.sample_bytes - sum(remaining)
    streams = [
        iter(_stream_source(source, revisions[source.name]))
        for source in config.data.sources
    ]

    active = True
    while active:
        active = False
        for index, (source, stream) in enumerate(zip(config.data.sources, streams)):
            if remaining[index] <= 0:
                continue
            active = True
            try:
                row = next(stream)
            except StopIteration as exc:
                raise RuntimeError(
                    f"source {source.name!r} exhausted before tokenizer sample target"
                ) from exc
            text = clean_document(row.get(source.text_field))
            if text is None:
                continue
            encoded = text.encode("utf-8")
            remaining[index] -= len(encoded)
            yield text


def train_tokenizer(config: PretrainingConfig, output_dir: Path) -> dict:
    from huggingface_hub import HfApi
    from tokenizers import Tokenizer, decoders, pre_tokenizers
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer
    from transformers import PreTrainedTokenizerFast

    output_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    revisions: dict[str, str] = {}
    for source in config.data.sources:
        revision = api.dataset_info(
            source.dataset,
            revision=source.revision,
        ).sha
        if revision is None:
            raise RuntimeError(f"could not resolve revision for {source.dataset}")
        revisions[source.name] = revision
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = BpeTrainer(
        vocab_size=config.tokenizer.vocab_size,
        min_frequency=2,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(
        _weighted_text_iterator(config, revisions),
        trainer=trainer,
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        unk_token=UNK_TOKEN,
        additional_special_tokens=list(ROLE_TOKENS),
        model_max_length=config.tokenizer.model_max_length,
        clean_up_tokenization_spaces=False,
    )
    fast.save_pretrained(output_dir)

    tokenizer_json = output_dir / "tokenizer.json"
    digest = hashlib.sha256(tokenizer_json.read_bytes()).hexdigest()
    manifest = {
        "format_version": 1,
        "vocab_size": len(fast),
        "sample_bytes": config.tokenizer.sample_bytes,
        "special_token_ids": {
            token: fast.convert_tokens_to_ids(token) for token in SPECIAL_TOKENS
        },
        "sha256": digest,
        "sources": [
            {
                "name": source.name,
                "dataset": source.dataset,
                "subset": source.subset,
                "resolved_revision": revisions[source.name],
                "weight": source.weight,
                "declared_license": source.declared_license,
            }
            for source in config.data.sources
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_tokenizer(path: str | Path):
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast.from_pretrained(path)


def encode_documents(
    tokenizer,
    texts: Iterable[str],
) -> Iterator[list[int]]:
    eos_id = tokenizer.eos_token_id
    for text in texts:
        yield [*tokenizer.encode(text, add_special_tokens=False), eos_id]
