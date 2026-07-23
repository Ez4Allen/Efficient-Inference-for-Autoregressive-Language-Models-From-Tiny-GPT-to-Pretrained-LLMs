from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch


@dataclass
class SpeculativeOutput:
    output_ids: torch.Tensor
    generated_token_ids: torch.Tensor

    draft_forward_calls: int
    target_forward_calls: int

    proposed_tokens: int
    accepted_draft_tokens: int
    speculative_rounds: int

    target_prefill_time_seconds: float
    total_time_seconds: float

    @property
    def acceptance_rate(self) -> float:
        if self.proposed_tokens == 0:
            return 0.0

        return self.accepted_draft_tokens / self.proposed_tokens


def _synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _crop_past_key_values(
    past_key_values: Any,
    sequence_length: int,
) -> Any:
    if past_key_values is None:
        return None

    if hasattr(past_key_values, "crop"):
        past_key_values.crop(sequence_length)
        return past_key_values

    cropped_layers = []

    for layer_cache in past_key_values:
        cropped_layer = tuple(
            state[:, :, :sequence_length, :]
            for state in layer_cache
        )
        cropped_layers.append(cropped_layer)

    return tuple(cropped_layers)


@torch.inference_mode()
def _draft_propose(
    draft_model: Any,
    input_ids: torch.Tensor,
    num_tokens: int,
    eos_token_id: int | None,
) -> tuple[torch.Tensor, int]:
    if num_tokens <= 0:
        raise ValueError("num_tokens must be greater than zero.")

    proposed_tokens: list[torch.Tensor] = []
    forward_calls = 0

    outputs = draft_model(
        input_ids=input_ids,
        use_cache=True,
    )
    forward_calls += 1

    draft_cache = outputs.past_key_values

    next_token = torch.argmax(
        outputs.logits[:, -1, :],
        dim=-1,
        keepdim=True,
    )

    proposed_tokens.append(next_token)

    if (
        eos_token_id is not None
        and next_token.item() == eos_token_id
    ):
        return torch.cat(proposed_tokens, dim=1), forward_calls

    for _ in range(num_tokens - 1):
        outputs = draft_model(
            input_ids=next_token,
            past_key_values=draft_cache,
            use_cache=True,
        )
        forward_calls += 1

        draft_cache = outputs.past_key_values

        next_token = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        proposed_tokens.append(next_token)

        if (
            eos_token_id is not None
            and next_token.item() == eos_token_id
        ):
            break

    return torch.cat(proposed_tokens, dim=1), forward_calls


@torch.inference_mode()
def greedy_speculative_decode(
    draft_model: Any,
    target_model: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    draft_tokens_per_round: int = 4,
    eos_token_id: int | None = None,
) -> SpeculativeOutput:
    if input_ids.ndim != 2:
        raise ValueError(
            "input_ids must have shape [batch_size, sequence_length]."
        )

    if input_ids.shape[0] != 1:
        raise ValueError(
            "This implementation currently supports batch size 1 only."
        )

    if max_new_tokens <= 0:
        raise ValueError(
            "max_new_tokens must be greater than zero."
        )

    if draft_tokens_per_round <= 0:
        raise ValueError(
            "draft_tokens_per_round must be greater than zero."
        )

    device = input_ids.device
    prompt_length = input_ids.shape[1]
    current_ids = input_ids.clone()

    draft_forward_calls = 0
    target_forward_calls = 0
    proposed_tokens = 0
    accepted_draft_tokens = 0
    speculative_rounds = 0

    _synchronize_if_needed(device)
    total_start = perf_counter()

    _synchronize_if_needed(device)
    prefill_start = perf_counter()

    target_outputs = target_model(
        input_ids=current_ids,
        use_cache=True,
    )
    target_forward_calls += 1

    _synchronize_if_needed(device)
    target_prefill_time_seconds = perf_counter() - prefill_start

    target_cache = target_outputs.past_key_values
    target_next_logits = target_outputs.logits[:, -1, :]

    while current_ids.shape[1] - prompt_length < max_new_tokens:
        speculative_rounds += 1

        generated_count = current_ids.shape[1] - prompt_length
        remaining_tokens = max_new_tokens - generated_count

        proposal_length = min(
            draft_tokens_per_round,
            remaining_tokens,
        )

        draft_proposals, round_draft_calls = _draft_propose(
            draft_model=draft_model,
            input_ids=current_ids,
            num_tokens=proposal_length,
            eos_token_id=eos_token_id,
        )

        draft_forward_calls += round_draft_calls

        actual_proposal_length = draft_proposals.shape[1]
        proposed_tokens += actual_proposal_length
        context_length = current_ids.shape[1]

        target_outputs = target_model(
            input_ids=draft_proposals,
            past_key_values=target_cache,
            use_cache=True,
        )
        target_forward_calls += 1

        verification_cache = target_outputs.past_key_values

        first_prediction = torch.argmax(
            target_next_logits,
            dim=-1,
            keepdim=True,
        )

        if actual_proposal_length > 1:
            remaining_predictions = torch.argmax(
                target_outputs.logits[:, :-1, :],
                dim=-1,
            )

            target_predictions = torch.cat(
                [first_prediction, remaining_predictions],
                dim=1,
            )
        else:
            target_predictions = first_prediction

        mismatch_positions = torch.nonzero(
            draft_proposals != target_predictions,
            as_tuple=False,
        )
        mismatch_index = (
            int(mismatch_positions[0, 1].item())
            if mismatch_positions.numel()
            else None
        )

        if mismatch_index is not None:
            if mismatch_index > 0:
                accepted_prefix = draft_proposals[
                    :,
                    :mismatch_index,
                ]

                current_ids = torch.cat(
                    [current_ids, accepted_prefix],
                    dim=1,
                )

                accepted_draft_tokens += mismatch_index

            accepted_context_length = (
                context_length + mismatch_index
            )

            target_cache = _crop_past_key_values(
                verification_cache,
                accepted_context_length,
            )

            correction_token = target_predictions[
                :,
                mismatch_index:
                mismatch_index + 1,
            ]

            current_ids = torch.cat(
                [current_ids, correction_token],
                dim=1,
            )

            if (
                eos_token_id is not None
                and correction_token.item() == eos_token_id
            ):
                break

            if current_ids.shape[1] - prompt_length >= max_new_tokens:
                break

            correction_outputs = target_model(
                input_ids=correction_token,
                past_key_values=target_cache,
                use_cache=True,
            )
            target_forward_calls += 1

            target_cache = correction_outputs.past_key_values
            target_next_logits = correction_outputs.logits[:, -1, :]

        else:
            current_ids = torch.cat(
                [current_ids, draft_proposals],
                dim=1,
            )

            accepted_draft_tokens += actual_proposal_length
            target_cache = verification_cache

            if (
                eos_token_id is not None
                and draft_proposals[0, -1].item() == eos_token_id
            ):
                break

            generated_count = current_ids.shape[1] - prompt_length

            if generated_count >= max_new_tokens:
                break

            bonus_token = torch.argmax(
                target_outputs.logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )

            current_ids = torch.cat(
                [current_ids, bonus_token],
                dim=1,
            )

            if (
                eos_token_id is not None
                and bonus_token.item() == eos_token_id
            ):
                break

            bonus_outputs = target_model(
                input_ids=bonus_token,
                past_key_values=target_cache,
                use_cache=True,
            )
            target_forward_calls += 1

            target_cache = bonus_outputs.past_key_values
            target_next_logits = bonus_outputs.logits[:, -1, :]

    current_ids = current_ids[
        :,
        :prompt_length + max_new_tokens,
    ]

    generated_token_ids = current_ids[:, prompt_length:]

    _synchronize_if_needed(device)
    total_time_seconds = perf_counter() - total_start

    return SpeculativeOutput(
        output_ids=current_ids,
        generated_token_ids=generated_token_ids,
        draft_forward_calls=draft_forward_calls,
        target_forward_calls=target_forward_calls,
        proposed_tokens=proposed_tokens,
        accepted_draft_tokens=accepted_draft_tokens,
        speculative_rounds=speculative_rounds,
        target_prefill_time_seconds=target_prefill_time_seconds,
        total_time_seconds=total_time_seconds,
    )