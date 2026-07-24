from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from src.inference.autoregressive import greedy_decode
from src.inference.speculative import greedy_speculative_decode


@dataclass
class ToyCache:
    tokens: torch.Tensor

    def crop(self, sequence_length: int) -> None:
        self.tokens = self.tokens[:, :sequence_length]


@dataclass
class ToyOutput:
    logits: torch.Tensor
    past_key_values: ToyCache


class ToyCausalLM:
    """Deterministic causal model used to exercise cache semantics."""

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
        logits = torch.full(
            (*input_ids.shape, self.vocab_size),
            -1_000.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        logits.scatter_(2, predictions.unsqueeze(-1), 1_000.0)
        return ToyOutput(logits=logits, past_key_values=ToyCache(full_tokens))


def test_greedy_decode_uses_cache_and_generates_expected_tokens() -> None:
    model = ToyCausalLM(offset=1)
    prompt = torch.tensor([[2, 5]], dtype=torch.long)

    result = greedy_decode(model, prompt, max_new_tokens=4)

    assert result.generated_token_ids.tolist() == [[6, 7, 8, 9]]
    assert result.target_forward_calls == 4
    assert len(result.decode_times_seconds) == 3


def test_speculative_decode_matches_baseline_with_perfect_draft() -> None:
    target = ToyCausalLM(offset=1)
    draft = ToyCausalLM(offset=1)
    prompt = torch.tensor([[3, 4]], dtype=torch.long)

    baseline = greedy_decode(target, prompt, max_new_tokens=9)
    speculative = greedy_speculative_decode(
        draft,
        target,
        prompt,
        max_new_tokens=9,
        draft_tokens_per_round=4,
    )

    assert torch.equal(
        baseline.generated_token_ids,
        speculative.generated_token_ids,
    )
    assert speculative.acceptance_rate == pytest.approx(1.0)
    assert speculative.accepted_draft_tokens == speculative.proposed_tokens


def test_speculative_decode_corrects_bad_draft() -> None:
    target = ToyCausalLM(offset=1)
    draft = ToyCausalLM(offset=2)
    prompt = torch.tensor([[1, 7]], dtype=torch.long)

    baseline = greedy_decode(target, prompt, max_new_tokens=7)
    speculative = greedy_speculative_decode(
        draft,
        target,
        prompt,
        max_new_tokens=7,
        draft_tokens_per_round=3,
    )

    assert torch.equal(
        baseline.generated_token_ids,
        speculative.generated_token_ids,
    )
    assert speculative.accepted_draft_tokens == 0
    assert speculative.acceptance_rate == 0.0


def test_decode_input_validation() -> None:
    model = ToyCausalLM()
    prompt = torch.tensor([[1, 2]], dtype=torch.long)

    with pytest.raises(ValueError):
        greedy_decode(model, prompt, max_new_tokens=0)

    with pytest.raises(ValueError):
        greedy_speculative_decode(
            model,
            model,
            prompt.repeat(2, 1),
            max_new_tokens=2,
        )


def test_greedy_decode_handles_per_sequence_eos_in_batches() -> None:
    model = ToyCausalLM(offset=1)
    prompt = torch.tensor([[1], [2]], dtype=torch.long)

    result = greedy_decode(
        model,
        prompt,
        max_new_tokens=4,
        eos_token_id=3,
    )

    assert result.generated_token_ids.tolist() == [[2, 3], [3, 3]]


def test_greedy_decode_accepts_multiple_eos_ids() -> None:
    model = ToyCausalLM(offset=1)
    prompt = torch.tensor([[1]], dtype=torch.long)

    result = greedy_decode(
        model,
        prompt,
        max_new_tokens=5,
        eos_token_id=[2, 9],
    )

    assert result.generated_token_ids.tolist() == [[2]]


def test_speculative_decode_accepts_multiple_eos_ids() -> None:
    model = ToyCausalLM(offset=1)
    prompt = torch.tensor([[1]], dtype=torch.long)

    result = greedy_speculative_decode(
        model,
        model,
        prompt,
        max_new_tokens=5,
        draft_tokens_per_round=3,
        eos_token_id=[2, 9],
    )

    assert result.generated_token_ids.tolist() == [[2]]
