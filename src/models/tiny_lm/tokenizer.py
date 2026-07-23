"""A minimal character-level tokenizer for the TinyGPT example."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


class CharTokenizer:
    """Map each character observed during fitting to an integer token ID."""

    def __init__(self) -> None:
        self.stoi: dict[str, int] | None = None
        self.itos: dict[int, str] | None = None
        self.vocab_size: int | None = None

    def fit(self, text: str) -> "CharTokenizer":
        if not isinstance(text, str):
            raise TypeError("text must be a string.")
        if not text:
            raise ValueError("Cannot fit a tokenizer on empty text.")
        characters = sorted(set(text))
        self.stoi = {character: index for index, character in enumerate(characters)}
        self.itos = {index: character for index, character in enumerate(characters)}
        self.vocab_size = len(characters)
        return self

    def encode(self, text: str) -> list[int]:
        if self.stoi is None:
            raise ValueError("Tokenizer has not been fitted yet.")
        unknown = sorted(set(text) - set(self.stoi))
        if unknown:
            preview = ", ".join(repr(character) for character in unknown[:8])
            raise ValueError(f"Text contains characters outside the vocabulary: {preview}")
        return [self.stoi[character] for character in text]

    def decode(self, ids: Iterable[int]) -> str:
        if self.itos is None:
            raise ValueError("Tokenizer has not been fitted yet.")
        characters: list[str] = []
        for raw_id in ids:
            token_id = int(raw_id)
            if token_id not in self.itos:
                raise ValueError(f"Unknown token ID: {token_id}")
            characters.append(self.itos[token_id])
        return "".join(characters)

    def save(self, path: str | Path) -> Path:
        if self.stoi is None or self.itos is None or self.vocab_size is None:
            raise ValueError("Tokenizer has not been fitted yet.")
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "stoi": self.stoi,
                    "itos": {str(key): value for key, value in self.itos.items()},
                    "vocab_size": self.vocab_size,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        source = Path(path).expanduser().resolve()
        data = json.loads(source.read_text(encoding="utf-8"))
        tokenizer = cls()
        tokenizer.stoi = {str(key): int(value) for key, value in data["stoi"].items()}
        tokenizer.itos = {int(key): str(value) for key, value in data["itos"].items()}
        tokenizer.vocab_size = int(data["vocab_size"])
        if len(tokenizer.stoi) != tokenizer.vocab_size:
            raise ValueError("Tokenizer vocabulary size does not match stoi.")
        if len(tokenizer.itos) != tokenizer.vocab_size:
            raise ValueError("Tokenizer vocabulary size does not match itos.")
        return tokenizer
