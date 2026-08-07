"""KV-cache container used by the custom draft model."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch


LayerCache = tuple[torch.Tensor, torch.Tensor]


@dataclass
class TinyQwenDraftCache:
    """A crop-able per-layer KV cache compatible with the project decoder.

    Each key/value tensor has shape ``[batch, kv_heads, sequence, head_dim]``.
    ``crop`` mutates the cache to match Hugging Face dynamic-cache semantics and
    the existing speculative-decoding helper in this repository.
    """

    layers: list[LayerCache]

    def __post_init__(self) -> None:
        for index, layer in enumerate(self.layers):
            if not isinstance(layer, tuple) or len(layer) != 2:
                raise TypeError(f"Cache layer {index} must be a (key, value) tuple.")
            key, value = layer
            if key.ndim != 4 or value.ndim != 4:
                raise ValueError("KV-cache tensors must be rank four.")
            if key.shape != value.shape:
                raise ValueError("Key and value cache tensors must have equal shapes.")

    def __len__(self) -> int:
        return len(self.layers)

    def __iter__(self) -> Iterator[LayerCache]:
        return iter(self.layers)

    def __getitem__(self, index: int) -> LayerCache:
        return self.layers[index]

    @property
    def sequence_length(self) -> int:
        if not self.layers:
            return 0
        return int(self.layers[0][0].shape[-2])

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if not self.layers:
            return 0
        return int(self.layers[layer_idx][0].shape[-2])

    def crop(self, sequence_length: int) -> None:
        if (
            isinstance(sequence_length, bool)
            or not isinstance(sequence_length, int)
            or sequence_length < 0
        ):
            raise ValueError("sequence_length must be a non-negative integer.")
        self.layers = [
            (
                key[:, :, :sequence_length, :],
                value[:, :, :sequence_length, :],
            )
            for key, value in self.layers
        ]

    def to_legacy_cache(self) -> tuple[LayerCache, ...]:
        return tuple(self.layers)

    @classmethod
    def from_legacy_cache(
        cls,
        cache: Sequence[Sequence[torch.Tensor]],
    ) -> "TinyQwenDraftCache":
        layers: list[LayerCache] = []
        for index, layer in enumerate(cache):
            if len(layer) != 2:
                raise ValueError(f"Cache layer {index} must contain key and value.")
            layers.append((layer[0], layer[1]))
        return cls(layers)
