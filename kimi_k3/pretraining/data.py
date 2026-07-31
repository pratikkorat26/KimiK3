"""Deterministic token-shard creation and memory-mapped packed datasets."""

from __future__ import annotations

import bisect
import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import PretrainingConfig, SourceConfig
from .tokenizer import clean_document, load_tokenizer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TokenShardWriter:
    def __init__(self, root: Path, shard_tokens: int) -> None:
        self.root = root
        self.shard_tokens = shard_tokens
        self.root.mkdir(parents=True, exist_ok=True)
        self.buffer: list[int] = []
        self.shards: list[dict[str, Any]] = []

    def append(self, ids: list[int], remaining: int) -> int:
        take = min(len(ids), remaining)
        self.buffer.extend(ids[:take])
        while len(self.buffer) >= self.shard_tokens:
            self._flush(self.shard_tokens)
        return take

    def finish(self) -> None:
        if self.buffer:
            self._flush(len(self.buffer))

    def _flush(self, count: int) -> None:
        shard_index = len(self.shards)
        path = self.root / f"shard-{shard_index:05d}.bin"
        values = np.asarray(self.buffer[:count], dtype="<u2")
        values.tofile(path)
        del self.buffer[:count]
        self.shards.append(
            {
                "path": path.as_posix(),
                "tokens": int(values.size),
                "sha256": _sha256(path),
            }
        )


class ProvenanceWriter:
    """Compressed document ledger without retaining source text."""

    _METADATA_FIELDS = (
        "license",
        "url",
        "repo_name",
        "path",
        "blob_id",
        "content_id",
        "language",
    )

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = gzip.open(path, mode="wt", encoding="utf-8")

    def append(
        self,
        *,
        row: dict[str, Any],
        source: SourceConfig,
        split: str,
        text: str,
        token_count: int,
    ) -> None:
        metadata = row.get("metadata")
        selected_metadata = {}
        if isinstance(metadata, dict):
            selected_metadata = {
                name: metadata[name]
                for name in self._METADATA_FIELDS
                if metadata.get(name) is not None
            }
        record = {
            "source": source.name,
            "dataset": source.dataset,
            "split": split,
            "document_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "token_count": token_count,
            "id": row.get("id"),
            "url": row.get("url"),
            "source_label": row.get("source"),
            "metadata": selected_metadata,
        }
        self.handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    def finish(self) -> dict[str, Any]:
        self.handle.close()
        return {
            "path": self.path.as_posix(),
            "sha256": _sha256(self.path),
        }


def _resolve_revision(source: SourceConfig) -> tuple[str, str | None]:
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(source.dataset, revision=source.revision)
    license_value = None
    if info.card_data is not None:
        license_value = info.card_data.get("license")
    if info.sha is None:
        raise RuntimeError(f"could not resolve revision for {source.dataset}")
    return info.sha, license_value


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


def _write_source_splits(
    *,
    rows,
    source: SourceConfig,
    tokenizer,
    writers: dict[str, TokenShardWriter],
    targets: dict[str, int],
    provenance: ProvenanceWriter,
) -> dict[str, dict[str, int]]:
    written = {"train": 0, "validation": 0}
    documents = {"train": 0, "validation": 0}
    validation_fraction = targets["validation"] / sum(targets.values())
    for row in rows:
        text = clean_document(row.get(source.text_field))
        if text is None:
            continue
        bucket_value = int.from_bytes(
            hashlib.sha256(f"{source.name}\0{text}".encode("utf-8")).digest()[:8],
            "big",
        ) / 2**64
        split = "validation" if bucket_value < validation_fraction else "train"
        if written[split] >= targets[split]:
            split = "train" if split == "validation" else "validation"
        if written[split] >= targets[split]:
            break
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids.append(tokenizer.eos_token_id)
        accepted = writers[split].append(
            ids,
            targets[split] - written[split],
        )
        written[split] += accepted
        documents[split] += 1
        provenance.append(
            row=row,
            source=source,
            split=split,
            text=text,
            token_count=accepted,
        )
        if all(written[name] >= targets[name] for name in targets):
            break
    for split in targets:
        if written[split] != targets[split]:
            raise RuntimeError(
                f"source {source.name!r} exhausted for {split} at "
                f"{written[split]:,}/{targets[split]:,} tokens"
            )
        writers[split].finish()
    return {
        split: {"tokens": written[split], "documents": documents[split]}
        for split in targets
    }


def prepare_token_shards(
    config: PretrainingConfig,
    *,
    tokenizer_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    tokenizer = load_tokenizer(tokenizer_dir)
    tokenizer_manifest = json.loads(
        (tokenizer_dir / "manifest.json").read_text(encoding="utf-8")
    )
    manifest: dict[str, Any] = {
        "format_version": 1,
        "tokenizer_sha256": tokenizer_manifest["sha256"],
        "train_tokens": config.data.train_tokens,
        "validation_tokens": config.data.validation_tokens,
        "shard_tokens": config.data.shard_tokens,
        "sources": [],
    }

    for source_index, source in enumerate(config.data.sources):
        resolved_revision, license_value = _resolve_revision(source)
        rows = iter(_stream_source(source, resolved_revision))
        source_entry: dict[str, Any] = {
            "name": source.name,
            "dataset": source.dataset,
            "subset": source.subset,
            "requested_revision": source.revision,
            "resolved_revision": resolved_revision,
            "dataset_card_license": license_value,
            "declared_license": source.declared_license,
            "weight": source.weight,
            "text_field": source.text_field,
            "splits": {},
        }
        targets = {
            "validation": config.source_validation_tokens(source_index),
            "train": config.source_train_tokens(source_index),
        }
        writers = {
            split: TokenShardWriter(
                output_dir / source.name / split,
                config.data.shard_tokens,
            )
            for split in targets
        }
        provenance = ProvenanceWriter(
            output_dir / source.name / "provenance.jsonl.gz"
        )
        split_stats = _write_source_splits(
            rows=rows,
            source=source,
            tokenizer=tokenizer,
            writers=writers,
            targets=targets,
            provenance=provenance,
        )
        for split in targets:
            source_entry["splits"][split] = {
                **split_stats[split],
                "shards": writers[split].shards,
            }
        source_entry["provenance"] = provenance.finish()
        manifest["sources"].append(source_entry)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


@dataclass(frozen=True)
class _Segment:
    path: Path
    token_offset: int
    sequence_length: int
    sequences: int


class PackedShardDataset(Dataset):
    """Map-style packed dataset over non-overlapping memory-mapped shard ranges."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str,
        sequence_length: int,
        source_ranges: dict[str, tuple[int, int]] | None = None,
        source_name: str | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        base = self.manifest_path.parent
        self.segments: list[_Segment] = []
        self.cumulative: list[int] = []
        total_sequences = 0

        for source in manifest["sources"]:
            name = source["name"]
            if source_name is not None and name != source_name:
                continue
            start, count = (
                source_ranges[name]
                if source_ranges is not None
                else (0, source["splits"][split]["tokens"])
            )
            stop = start + count
            cursor = 0
            for shard in source["splits"][split]["shards"]:
                shard_start, shard_stop = cursor, cursor + shard["tokens"]
                cursor = shard_stop
                overlap_start = max(start, shard_start)
                overlap_stop = min(stop, shard_stop)
                usable = overlap_stop - overlap_start
                sequences = usable // sequence_length
                if sequences <= 0:
                    continue
                path = Path(shard["path"])
                if not path.is_absolute():
                    path = base / path
                segment = _Segment(
                    path=path,
                    token_offset=overlap_start - shard_start,
                    sequence_length=sequence_length,
                    sequences=sequences,
                )
                self.segments.append(segment)
                total_sequences += sequences
                self.cumulative.append(total_sequences)
        if total_sequences == 0:
            raise ValueError("packed dataset contains no complete sequences")
        self._maps: dict[Path, np.memmap] = {}

    def __len__(self) -> int:
        return self.cumulative[-1]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        segment_index = bisect.bisect_right(self.cumulative, index)
        prior = self.cumulative[segment_index - 1] if segment_index else 0
        segment = self.segments[segment_index]
        local = index - prior
        offset = segment.token_offset + local * segment.sequence_length
        values = self._maps.get(segment.path)
        if values is None:
            values = np.memmap(segment.path, mode="r", dtype="<u2")
            self._maps[segment.path] = values
        ids = torch.from_numpy(
            np.asarray(
                values[offset : offset + segment.sequence_length],
                dtype=np.int64,
            )
        )
        return {"input_ids": ids, "labels": ids.clone()}


def curriculum_source_ranges(
    config: PretrainingConfig,
    stage_index: int,
) -> dict[str, tuple[int, int]]:
    ranges: dict[str, tuple[int, int]] = {}
    for source_index, source in enumerate(config.data.sources):
        start = sum(
            config.stage_source_tokens(prior, source_index)
            for prior in range(stage_index)
        )
        ranges[source.name] = (
            start,
            config.stage_source_tokens(stage_index, source_index),
        )
    return ranges
