"""Sampling helpers shared by the tiny language model."""

from __future__ import annotations

import torch


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one token ID from ``[batch, vocab]`` logits."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch_size, vocab_size].")
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero.")
    scaled = logits / temperature
    if top_k is not None:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer or None.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        top_k = min(top_k, scaled.shape[-1])
        threshold = torch.topk(scaled, top_k, dim=-1).values[:, -1:]
        scaled = scaled.masked_fill(scaled < threshold, float("-inf"))
    probabilities = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator)
