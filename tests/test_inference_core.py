from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from src.inference.autoregressive import greedy_decode, sample_decode
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



def test_sample_decode_is_reproducible_with_seeded_generator() -> None:
    model = ToyCausalLM(offset=1)
    prompt = torch.tensor([[2, 5]], dtype=torch.long)

    first_generator = torch.Generator(device=prompt.device).manual_seed(123)
    second_generator = torch.Generator(device=prompt.device).manual_seed(123)

    first = sample_decode(
        model,
        prompt,
        max_new_tokens=4,
        temperature=0.8,
        top_p=0.9,
        generator=first_generator,
    )
    second = sample_decode(
        model,
        prompt,
        max_new_tokens=4,
        temperature=0.8,
        top_p=0.9,
        generator=second_generator,
    )

    assert torch.equal(first.generated_token_ids, second.generated_token_ids)
    assert first.generated_token_ids.tolist() == [[6, 7, 8, 9]]


def test_greedy_decode_uses_cache_and_generates_expected_tokens() -> None:
    model = ToyCausalLM(offset=1)
    prompt = torch.tensor([[2, 5]], dtype=torch.long)

    result = greedy_decode(model, prompt, max_new_tokens=4)

    assert result.generated_token_ids.tolist() == [[6, 7, 8, 9]]
    assert result.target_forward_calls == 4
    assert len(result.decode_times_seconds) == 3


@pytest.mark.parametrize("verification_mode", ["exact", "block"])
def test_speculative_decode_matches_baseline_with_perfect_draft(
    verification_mode: str,
) -> None:
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
        verification_mode=verification_mode,
    )

    assert speculative.verification_mode == verification_mode
    assert torch.equal(
        baseline.generated_token_ids,
        speculative.generated_token_ids,
    )
    assert speculative.acceptance_rate == pytest.approx(1.0)
    assert speculative.accepted_draft_tokens == speculative.proposed_tokens


@pytest.mark.parametrize("verification_mode", ["exact", "block"])
def test_speculative_decode_corrects_bad_draft(
    verification_mode: str,
) -> None:
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
        verification_mode=verification_mode,
    )

    assert speculative.verification_mode == verification_mode
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


def test_speculative_decode_prefills_draft_once_and_reuses_cache() -> None:
    class RecordingToyCausalLM(ToyCausalLM):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.calls: list[tuple[int, bool]] = []

        def __call__(self, **kwargs):
            input_ids = kwargs["input_ids"]
            self.calls.append(
                (int(input_ids.shape[1]), kwargs.get("past_key_values") is not None)
            )
            return super().__call__(**kwargs)

    draft = RecordingToyCausalLM(offset=1)
    target = ToyCausalLM(offset=1)
    prompt = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)

    result = greedy_speculative_decode(
        draft,
        target,
        prompt,
        max_new_tokens=11,
        draft_tokens_per_round=4,
    )

    prompt_prefills = [call for call in draft.calls if call == (6, False)]
    assert len(prompt_prefills) == 1
    assert all(length == 1 for length, _ in draft.calls[1:])
    assert all(has_cache for _, has_cache in draft.calls[1:])
    assert result.draft_prefill_time_seconds >= 0.0
    assert result.time_to_first_token_seconds >= result.prefill_time_seconds

@pytest.mark.parametrize("verification_mode", ["exact", "block"])
def test_speculative_decode_recovers_from_middle_block_mismatch(
    verification_mode: str,
) -> None:
    class PrefixSumCausalLM:
        def __init__(
            self,
            *,
            vocab_size: int = 31,
            perturb_at_length: int | None = None,
        ) -> None:
            self.vocab_size = vocab_size
            self.perturb_at_length = perturb_at_length

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
            prefix_sum = prefix.sum(dim=1, keepdim=True)
            running_sum = prefix_sum + input_ids.cumsum(dim=1)
            predictions = (running_sum + 1) % self.vocab_size

            if self.perturb_at_length is not None:
                total_positions = (
                    torch.arange(
                        1,
                        input_ids.shape[1] + 1,
                        device=input_ids.device,
                    )
                    + prefix.shape[1]
                )
                perturb = total_positions == self.perturb_at_length
                predictions = torch.where(
                    perturb.unsqueeze(0),
                    (predictions + 1) % self.vocab_size,
                    predictions,
                )

            logits = torch.full(
                (*input_ids.shape, self.vocab_size),
                -1_000.0,
                dtype=torch.float32,
                device=input_ids.device,
            )
            logits.scatter_(2, predictions.unsqueeze(-1), 1_000.0)
            return ToyOutput(logits=logits, past_key_values=ToyCache(full_tokens))

    target = PrefixSumCausalLM()
    # Prompt length is three. The draft's first proposal is correct, but its
    # prediction after consuming that proposal (context length four) is wrong.
    # Correct cache cropping lets the draft realign after target correction.
    draft = PrefixSumCausalLM(perturb_at_length=4)
    prompt = torch.tensor([[2, 4, 6]], dtype=torch.long)

    baseline = greedy_decode(target, prompt, max_new_tokens=12)
    speculative = greedy_speculative_decode(
        draft,
        target,
        prompt,
        max_new_tokens=12,
        draft_tokens_per_round=4,
        verification_mode=verification_mode,
    )

    assert speculative.verification_mode == verification_mode
    assert torch.equal(
        baseline.generated_token_ids,
        speculative.generated_token_ids,
    )
    assert 0 < speculative.accepted_draft_tokens < speculative.proposed_tokens
    assert speculative.acceptance_rate > 0.5


def test_exact_verification_handles_query_length_sensitive_target() -> None:
    class QueryLengthSensitiveCausalLM(ToyCausalLM):
        def __call__(self, **kwargs):
            output = super().__call__(**kwargs)
            input_ids = kwargs["input_ids"]
            has_cache = kwargs.get("past_key_values") is not None
            if has_cache and input_ids.shape[1] > 1:
                # Simulate a low-precision target kernel whose q_len>1 block
                # path flips one argmax while q_len=1 greedy decode is stable.
                predictions = torch.argmax(output.logits, dim=-1)
                predictions[:, 0] = (predictions[:, 0] + 1) % self.vocab_size
                logits = torch.full_like(output.logits, -1_000.0)
                logits.scatter_(2, predictions.unsqueeze(-1), 1_000.0)
                output = ToyOutput(logits=logits, past_key_values=output.past_key_values)
            return output

    prompt = torch.tensor([[3, 4]], dtype=torch.long)
    baseline = greedy_decode(ToyCausalLM(offset=1), prompt, max_new_tokens=9)

    block = greedy_speculative_decode(
        ToyCausalLM(offset=1),
        QueryLengthSensitiveCausalLM(offset=1),
        prompt,
        max_new_tokens=9,
        draft_tokens_per_round=4,
        verification_mode="block",
    )
    exact = greedy_speculative_decode(
        ToyCausalLM(offset=1),
        QueryLengthSensitiveCausalLM(offset=1),
        prompt,
        max_new_tokens=9,
        draft_tokens_per_round=4,
        verification_mode="exact",
    )

    assert not torch.equal(block.generated_token_ids, baseline.generated_token_ids)
    assert torch.equal(exact.generated_token_ids, baseline.generated_token_ids)
    assert exact.verification_mode == "exact"
    assert exact.target_forward_calls == baseline.target_forward_calls


def test_speculative_decode_rejects_unknown_verification_mode() -> None:
    model = ToyCausalLM()
    prompt = torch.tensor([[1, 2]], dtype=torch.long)

    with pytest.raises(ValueError, match="verification_mode"):
        greedy_speculative_decode(
            model,
            model,
            prompt,
            max_new_tokens=2,
            verification_mode="magic",
        )
