from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from src.inference.autoregressive import greedy_decode
from src.inference.consistency import diagnose_target_block_consistency


@dataclass
class ToyCache:
    tokens: torch.Tensor

    def crop(self, sequence_length: int) -> None:
        self.tokens = self.tokens[:, :sequence_length]


@dataclass
class ToyOutput:
    logits: torch.Tensor
    past_key_values: ToyCache


class QueryLengthSensitiveModel:
    def __init__(self, *, vocab_size: int = 17, offset: int = 1) -> None:
        self.vocab_size = vocab_size
        self.offset = offset

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        past_key_values: ToyCache | None = None,
        use_cache: bool = True,
    ) -> ToyOutput:
        del use_cache
        prefix = (
            past_key_values.tokens
            if past_key_values is not None
            else input_ids[:, :0]
        )
        full_tokens = torch.cat([prefix, input_ids], dim=1)
        predictions = (input_ids + self.offset) % self.vocab_size

        if past_key_values is not None and input_ids.shape[1] > 1:
            predictions = predictions.clone()
            predictions[:, 0] = (
                predictions[:, 0] + 1
            ) % self.vocab_size

        logits = torch.full(
            (*input_ids.shape, self.vocab_size),
            -1_000.0,
            dtype=torch.float32,
        )
        logits.scatter_(2, predictions.unsqueeze(-1), 1_000.0)
        return ToyOutput(logits=logits, past_key_values=ToyCache(full_tokens))


def test_consistency_report_detects_block_only_divergence() -> None:
    model = QueryLengthSensitiveModel()
    prompt = torch.tensor([[2, 5]], dtype=torch.long)
    reference = greedy_decode(model, prompt, max_new_tokens=8)

    sequential = diagnose_target_block_consistency(
        model,
        prompt,
        reference.generated_token_ids,
        block_size=1,
    )
    blocked = diagnose_target_block_consistency(
        model,
        prompt,
        reference.generated_token_ids,
        block_size=4,
    )

    assert sequential.exact_match
    assert sequential.mismatch_count == 0
    assert not blocked.exact_match
    assert blocked.mismatch_count > 0
    assert blocked.mismatches[0].position >= 1
    assert blocked.mismatches[0].predicted_margin > 0.0


def test_consistency_report_validates_inputs() -> None:
    model = QueryLengthSensitiveModel()
    prompt = torch.tensor([[2, 5]], dtype=torch.long)
    reference = torch.tensor([[6, 7]], dtype=torch.long)

    with pytest.raises(ValueError, match="block_size"):
        diagnose_target_block_consistency(
            model,
            prompt,
            reference,
            block_size=0,
        )
