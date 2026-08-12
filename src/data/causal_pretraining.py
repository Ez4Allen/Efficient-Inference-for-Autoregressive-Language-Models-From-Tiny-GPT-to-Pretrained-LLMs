"""Packed causal-language-model data for TinyQwenStudent pretraining."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.utils.io import read_jsonl

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
else:
    PreTrainedTokenizerBase = Any

IGNORE_INDEX = -100


@dataclass(slots=True)
class CausalCorpusRecord:
    record_id: str
    split: str
    language: str
    domain: str
    source_type: str
    text: str
    source_id: str | None = None
    license_name: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, index: int) -> "CausalCorpusRecord":
        record_id = str(value.get("id") or value.get("record_id") or f"record_{index:08d}").strip()
        split = str(value.get("split") or "train").strip().casefold()
        language = str(value.get("language") or "unknown").strip().casefold()
        domain = str(value.get("domain") or value.get("game") or "general").strip().casefold()
        source_type = str(value.get("source_type") or "text").strip().casefold()
        text = str(value.get("text") or "").strip()
        if not record_id:
            raise ValueError(f"Corpus record {index} has an empty id.")
        if split not in {"train", "validation"}:
            raise ValueError(
                f"Corpus record {record_id} has unsupported split {split!r}."
            )
        if not text:
            raise ValueError(f"Corpus record {record_id} has empty text.")
        return cls(
            record_id=record_id,
            split=split,
            language=language,
            domain=domain,
            source_type=source_type,
            text=text,
            source_id=(str(value.get("source_id")).strip() if value.get("source_id") else None),
            license_name=(
                str(value.get("license_name")).strip()
                if value.get("license_name")
                else None
            ),
        )


def load_causal_corpus(
    path: str | Path,
    *,
    split: str,
) -> list[CausalCorpusRecord]:
    normalized_split = str(split).strip().casefold()
    if normalized_split not in {"train", "validation"}:
        raise ValueError("split must be train or validation.")
    records = [
        CausalCorpusRecord.from_mapping(record, index=index)
        for index, record in enumerate(read_jsonl(path), start=1)
    ]
    selected = [record for record in records if record.split == normalized_split]
    if not selected:
        raise ValueError(f"No {normalized_split} records found in {path}.")
    return selected


def _encode_text(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        encoded = encode(text, add_special_tokens=False)
    else:
        encoded = tokenizer(text, add_special_tokens=False)
        if isinstance(encoded, Mapping):
            encoded = encoded.get("input_ids")
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.detach().cpu().tolist()
    if isinstance(encoded, list) and len(encoded) == 1 and isinstance(encoded[0], list):
        encoded = encoded[0]
    if not isinstance(encoded, list) or any(not isinstance(item, int) for item in encoded):
        raise TypeError("Tokenizer must return a flat list of integer token IDs.")
    return [int(item) for item in encoded]


class CausalPackedDataset(Dataset[dict[str, torch.Tensor]]):
    """Tokenize documents and pack them into causal-LM sequences.

    Documents are separated with EOS.  Splits are selected before tokenization,
    and no example from the formal evaluation files is loaded by this dataset.
    The final short chunk is retained when it contains at least two tokens.
    """

    def __init__(
        self,
        source: str | Path | Iterable[CausalCorpusRecord],
        tokenizer: PreTrainedTokenizerBase,
        *,
        split: str,
        max_length: int,
        stride: int | None = None,
        add_bos: bool = False,
    ) -> None:
        if max_length < 2:
            raise ValueError("max_length must be at least 2.")
        stride = max_length if stride is None else int(stride)
        if stride < 1 or stride > max_length:
            raise ValueError("stride must be in [1, max_length].")

        if isinstance(source, (str, Path)):
            records = load_causal_corpus(source, split=split)
        else:
            normalized_split = str(split).strip().casefold()
            records = [record for record in source if record.split == normalized_split]
            if not records:
                raise ValueError(f"No {normalized_split} corpus records were supplied.")

        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        bos_token_id = getattr(tokenizer, "bos_token_id", None)
        if eos_token_id is None:
            raise ValueError("Causal pretraining tokenizer must define eos_token_id.")
        if add_bos and bos_token_id is None:
            raise ValueError("add_bos=True requires tokenizer.bos_token_id.")

        stream: list[int] = []
        document_token_counts: list[int] = []
        for record in records:
            tokens = _encode_text(tokenizer, record.text)
            if add_bos:
                tokens.insert(0, int(bos_token_id))
            if not tokens or tokens[-1] != int(eos_token_id):
                tokens.append(int(eos_token_id))
            stream.extend(tokens)
            document_token_counts.append(len(tokens))

        chunks: list[torch.Tensor] = []
        for start in range(0, len(stream), stride):
            token_ids = stream[start : start + max_length]
            if len(token_ids) < 2:
                break
            chunks.append(torch.tensor(token_ids, dtype=torch.long))
            if start + max_length >= len(stream):
                break

        if not chunks:
            raise ValueError("Causal corpus produced no trainable chunks.")

        self.records = records
        self.chunks = chunks
        self.max_length = int(max_length)
        self.stride = int(stride)
        self.total_tokens = len(stream)
        self.document_token_counts = document_token_counts

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        input_ids = self.chunks[index].clone()
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids.clone(),
        }


class CausalDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            raise ValueError("Tokenizer must define PAD or EOS for batching.")
        self.pad_token_id = int(pad_token_id)

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("Cannot collate an empty batch.")
        input_ids = pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        attention_mask = pad_sequence(
            [item["attention_mask"] for item in batch],
            batch_first=True,
            padding_value=0,
        )
        labels = pad_sequence(
            [item["labels"] for item in batch],
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
