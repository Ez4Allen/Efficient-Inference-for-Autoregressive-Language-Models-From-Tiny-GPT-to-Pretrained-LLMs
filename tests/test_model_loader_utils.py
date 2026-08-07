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


def test_tiny_draft_tokenizer_contract_hash_and_vocab_size() -> None:
    from src.models.tokenizer_contract import (
        tokenizer_sha256,
        tokenizer_vocabulary_size,
        validate_model_tokenizer_contract,
    )

    tokenizer = FakeTokenizer({"a": 0, "b": 1, "<eos>": 2, "<unk>": 3})
    digest = tokenizer_sha256(tokenizer)
    assert len(digest) == 64
    assert tokenizer_vocabulary_size(tokenizer) == 4

    model = SimpleNamespace(
        config=SimpleNamespace(
            vocab_size=4,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            unk_token_id=3,
            tokenizer_sha256=digest,
            chat_template_sha256=None,
        )
    )
    validate_model_tokenizer_contract(model, tokenizer)

    model.config.tokenizer_sha256 = "0" * 64
    with pytest.raises(ValueError, match="Tokenizer mapping"):
        validate_model_tokenizer_contract(model, tokenizer)


def test_local_custom_checkpoint_prefers_saved_tokenizer_files(tmp_path) -> None:
    from src.models.loader import _contains_tokenizer_files

    assert not _contains_tokenizer_files(tmp_path)
    (tmp_path / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    assert _contains_tokenizer_files(tmp_path)
