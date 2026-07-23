from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.models.loader import (
    ModelBundle,
    resolve_device,
    resolve_dtype,
    validate_tokenizer_compatibility,
)


class FakeTokenizer:
    def __init__(self, vocab, *, eos=2, pad=0):
        self._vocab = vocab
        self.bos_token_id = 1
        self.eos_token_id = eos
        self.pad_token_id = pad
        self.unk_token_id = 3

    def get_vocab(self):
        return self._vocab


def bundle(tokenizer):
    return ModelBundle(
        model_name="fake",
        model=SimpleNamespace(),
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_device_and_dtype_resolution_on_cpu() -> None:
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_dtype(torch.device("cpu")) == torch.float32
    assert resolve_dtype(torch.device("cpu"), torch.float64) == torch.float64


def test_tokenizer_compatibility() -> None:
    first = bundle(FakeTokenizer({"a": 0, "b": 1}))
    second = bundle(FakeTokenizer({"a": 0, "b": 1}))
    validate_tokenizer_compatibility(first, second)

    mismatched = bundle(FakeTokenizer({"a": 0, "c": 1}))
    with pytest.raises(ValueError, match="different vocabularies"):
        validate_tokenizer_compatibility(first, mismatched)
