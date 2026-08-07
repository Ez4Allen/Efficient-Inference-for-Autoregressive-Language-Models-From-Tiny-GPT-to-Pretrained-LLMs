"""Exact tokenizer-contract utilities for speculative model pairs."""

from __future__ import annotations

import hashlib
from typing import Any


_SPECIAL_TOKEN_ATTRIBUTES = (
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "unk_token_id",
)


def tokenizer_vocabulary_size(tokenizer: Any) -> int:
    """Return the embedding size required by the tokenizer's largest token ID."""

    vocabulary = tokenizer.get_vocab()
    if not isinstance(vocabulary, dict) or not vocabulary:
        raise ValueError("Tokenizer get_vocab() must return a non-empty dictionary.")
    token_ids = list(vocabulary.values())
    if any(isinstance(value, bool) or not isinstance(value, int) for value in token_ids):
        raise TypeError("Tokenizer vocabulary IDs must be integers.")
    if min(token_ids) < 0:
        raise ValueError("Tokenizer vocabulary IDs must be non-negative.")
    return max(token_ids) + 1


def tokenizer_sha256(tokenizer: Any) -> str:
    """Hash the exact token-to-ID mapping and special-token IDs."""

    vocabulary = tokenizer.get_vocab()
    if not isinstance(vocabulary, dict) or not vocabulary:
        raise ValueError("Tokenizer get_vocab() must return a non-empty dictionary.")

    digest = hashlib.sha256()
    for token, token_id in sorted(
        vocabulary.items(),
        key=lambda item: (int(item[1]), str(item[0])),
    ):
        digest.update(str(int(token_id)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(token).encode("utf-8"))
        digest.update(b"\n")

    digest.update(b"special-tokens\n")
    for attribute in _SPECIAL_TOKEN_ATTRIBUTES:
        value = getattr(tokenizer, attribute, None)
        digest.update(attribute.encode("ascii"))
        digest.update(b"=")
        digest.update(str(value).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def chat_template_sha256(tokenizer: Any) -> str | None:
    template = getattr(tokenizer, "chat_template", None)
    if template is None:
        return None
    return hashlib.sha256(str(template).encode("utf-8")).hexdigest()


def validate_model_tokenizer_contract(model: Any, tokenizer: Any) -> None:
    """Validate vocabulary dimensions, IDs, and recorded tokenizer hashes."""

    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("Model does not expose a config object.")

    expected_vocab_size = int(getattr(config, "vocab_size", 0) or 0)
    actual_vocab_size = tokenizer_vocabulary_size(tokenizer)
    if expected_vocab_size != actual_vocab_size:
        raise ValueError(
            "Model/tokenizer vocabulary-size mismatch: "
            f"{expected_vocab_size} != {actual_vocab_size}."
        )

    for attribute in _SPECIAL_TOKEN_ATTRIBUTES:
        expected = getattr(config, attribute, None)
        actual = getattr(tokenizer, attribute, None)
        if expected is not None and expected != actual:
            raise ValueError(
                f"Model/tokenizer mismatch for {attribute}: {expected} != {actual}."
            )

    expected_hash = getattr(config, "tokenizer_sha256", None)
    if expected_hash is not None:
        actual_hash = tokenizer_sha256(tokenizer)
        if str(expected_hash).casefold() != actual_hash:
            raise ValueError(
                "Tokenizer mapping does not match the tokenizer used to train "
                "this draft checkpoint."
            )

    expected_template_hash = getattr(config, "chat_template_sha256", None)
    if expected_template_hash is not None:
        actual_template_hash = chat_template_sha256(tokenizer)
        if actual_template_hash != str(expected_template_hash).casefold():
            raise ValueError(
                "Tokenizer chat template does not match the draft checkpoint."
            )
