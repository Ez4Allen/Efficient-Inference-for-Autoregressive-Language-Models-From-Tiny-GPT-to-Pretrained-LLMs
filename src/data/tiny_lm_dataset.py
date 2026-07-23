"""Character-level next-token datasets for TinyGPT."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Return fixed-length ``(input, next-token target)`` windows."""

    def __init__(self, token_ids: Sequence[int] | torch.Tensor, block_size: int):
        if isinstance(block_size, bool) or not isinstance(block_size, int):
            raise TypeError("block_size must be an integer.")
        if block_size < 1:
            raise ValueError("block_size must be at least 1.")
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids, dtype=torch.long)
        else:
            token_ids = token_ids.to(dtype=torch.long)
        if token_ids.dim() != 1:
            raise ValueError("token_ids must be a 1D sequence of token IDs.")
        if len(token_ids) <= block_size:
            raise ValueError("token_ids length must be greater than block_size.")
        self.token_ids = token_ids
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.token_ids) - self.block_size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self.token_ids[index : index + self.block_size]
        targets = self.token_ids[index + 1 : index + self.block_size + 1]
        return inputs, targets


def create_train_val_split(
    token_ids: Sequence[int] | torch.Tensor,
    val_ratio: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the token stream contiguously to avoid cross-split leakage."""

    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")
    if not isinstance(token_ids, torch.Tensor):
        token_ids = torch.tensor(token_ids, dtype=torch.long)
    else:
        token_ids = token_ids.to(dtype=torch.long)
    if token_ids.ndim != 1:
        raise ValueError("token_ids must be one-dimensional.")
    if len(token_ids) < 2:
        raise ValueError("At least two token IDs are required.")

    validation_size = max(1, int(len(token_ids) * val_ratio))
    training_size = len(token_ids) - validation_size
    if training_size < 1:
        raise ValueError("Validation split leaves no training tokens.")
    return token_ids[:training_size], token_ids[training_size:]


def load_text_file(path: str | Path) -> str:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Text file not found: {source}")
    text = source.read_text(encoding="utf-8")
    if not text:
        raise ValueError(f"Text file is empty: {source}")
    return text
