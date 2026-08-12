from __future__ import annotations

from pathlib import Path

import torch

from src.data.causal_pretraining import CausalDataCollator, CausalPackedDataset


class TinyTokenizer:
    eos_token_id = 2
    bos_token_id = 1
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [3 + (ord(character) % 7) for character in text]


def test_causal_packed_dataset_and_collator(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '\n'.join(
            [
                '{"id":"a","split":"train","language":"en","domain":"x","source_type":"guide","text":"abcdef"}',
                '{"id":"b","split":"validation","language":"en","domain":"x","source_type":"guide","text":"uvwxyz"}',
            ]
        )
        + '\n',
        encoding="utf-8",
    )
    tokenizer = TinyTokenizer()
    dataset = CausalPackedDataset(
        corpus,
        tokenizer,
        split="train",
        max_length=4,
        stride=4,
    )
    assert len(dataset) >= 1
    item = dataset[0]
    assert torch.equal(item["input_ids"], item["labels"])

    batch = CausalDataCollator(tokenizer)([dataset[0], dataset[-1]])
    assert batch["input_ids"].ndim == 2
    assert batch["labels"].shape == batch["input_ids"].shape
