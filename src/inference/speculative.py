from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

import torch


SpeculativeVerificationMode = Literal["exact", "block"]


@dataclass
class SpeculativeOutput:
    output_ids: torch.Tensor
    generated_token_ids: torch.Tensor

    draft_forward_calls: int
    target_forward_calls: int

    proposed_tokens: int
    accepted_draft_tokens: int
    speculative_rounds: int

    draft_prefill_time_seconds: float
    target_prefill_time_seconds: float
    time_to_first_token_seconds: float
    total_time_seconds: float
    verification_mode: str = "exact"

    @property
    def acceptance_rate(self) -> float:
        if self.proposed_tokens == 0:
            return 0.0
        return self.accepted_draft_tokens / self.proposed_tokens

    @property
    def prefill_time_seconds(self) -> float:
        return self.draft_prefill_time_seconds + self.target_prefill_time_seconds

    @property
    def average_accepted_tokens_per_round(self) -> float:
        if self.speculative_rounds == 0:
            return 0.0
        return self.accepted_draft_tokens / self.speculative_rounds


@dataclass
class _PairPrefill:
    draft_cache: Any
    draft_next_logits: torch.Tensor
    target_cache: Any
    target_next_logits: torch.Tensor
    draft_forward_calls: int
    target_forward_calls: int
    draft_prefill_time_seconds: float
    target_prefill_time_seconds: float


def _synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _normalize_eos_token_ids(
    eos_token_id: int | Sequence[int] | None,
) -> tuple[int, ...]:
    if eos_token_id is None:
        return ()
    if isinstance(eos_token_id, int):
        return (eos_token_id,)
    result = tuple(dict.fromkeys(int(value) for value in eos_token_id))
    if not result:
        raise ValueError("eos_token_id sequence cannot be empty.")
    return result


def _normalize_verification_mode(value: str) -> SpeculativeVerificationMode:
    normalized = str(value).strip().casefold()
    if normalized not in {"exact", "block"}:
        raise ValueError("verification_mode must be 'exact' or 'block'.")
    return normalized  # type: ignore[return-value]


def _is_eos(token: torch.Tensor, eos_token_ids: tuple[int, ...]) -> bool:
    return bool(eos_token_ids) and int(token.item()) in eos_token_ids


def _crop_past_key_values(
    past_key_values: Any,
    sequence_length: int,
) -> Any:
    if past_key_values is None:
        return None
    if sequence_length < 0:
        raise ValueError("sequence_length cannot be negative.")

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


def _validate_inputs(
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    draft_tokens_per_round: int,
) -> None:
    if input_ids.ndim != 2:
        raise ValueError(
            "input_ids must have shape [batch_size, sequence_length]."
        )
    if input_ids.shape[0] != 1:
        raise ValueError(
            "This implementation currently supports batch size 1 only."
        )
    if input_ids.shape[1] < 1:
        raise ValueError("input_ids must contain at least one prompt token.")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than zero.")
    if draft_tokens_per_round <= 0:
        raise ValueError("draft_tokens_per_round must be greater than zero.")


@torch.inference_mode()
def _draft_propose(
    draft_model: Any,
    *,
    draft_cache: Any,
    draft_next_logits: torch.Tensor,
    num_tokens: int,
    eos_token_id: int | Sequence[int] | None,
) -> tuple[torch.Tensor, Any, torch.Tensor, int]:
    """Generate candidates from a persistent draft cache.

    The returned cache contains every proposal except the final proposal. The
    caller consumes that last token only after target verification. This avoids
    an unnecessary draft forward pass when the target rejects earlier in the
    candidate block while preserving rollback correctness.
    """

    if num_tokens <= 0:
        raise ValueError("num_tokens must be greater than zero.")

    eos_token_ids = _normalize_eos_token_ids(eos_token_id)
    proposed_tokens: list[torch.Tensor] = []
    forward_calls = 0
    cache = draft_cache
    next_logits = draft_next_logits

    for proposal_index in range(num_tokens):
        next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        proposed_tokens.append(next_token)

        if _is_eos(next_token, eos_token_ids):
            break
        if proposal_index == num_tokens - 1:
            break

        outputs = draft_model(
            input_ids=next_token,
            past_key_values=cache,
            use_cache=True,
        )
        forward_calls += 1
        cache = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]

    return torch.cat(proposed_tokens, dim=1), cache, next_logits, forward_calls


@torch.inference_mode()
def _advance_cache(
    model: Any,
    *,
    token: torch.Tensor,
    cache: Any,
) -> tuple[Any, torch.Tensor]:
    outputs = model(
        input_ids=token,
        past_key_values=cache,
        use_cache=True,
    )
    return outputs.past_key_values, outputs.logits[:, -1, :]


@torch.inference_mode()
def _prefill_pair(
    draft_model: Any,
    target_model: Any,
    input_ids: torch.Tensor,
) -> _PairPrefill:
    device = input_ids.device

    _synchronize_if_needed(device)
    draft_prefill_start = perf_counter()
    draft_outputs = draft_model(input_ids=input_ids, use_cache=True)
    _synchronize_if_needed(device)
    draft_prefill_time_seconds = perf_counter() - draft_prefill_start

    _synchronize_if_needed(device)
    target_prefill_start = perf_counter()
    target_outputs = target_model(input_ids=input_ids, use_cache=True)
    _synchronize_if_needed(device)
    target_prefill_time_seconds = perf_counter() - target_prefill_start

    return _PairPrefill(
        draft_cache=draft_outputs.past_key_values,
        draft_next_logits=draft_outputs.logits[:, -1, :],
        target_cache=target_outputs.past_key_values,
        target_next_logits=target_outputs.logits[:, -1, :],
        draft_forward_calls=1,
        target_forward_calls=1,
        draft_prefill_time_seconds=draft_prefill_time_seconds,
        target_prefill_time_seconds=target_prefill_time_seconds,
    )


def _finalize_output(
    *,
    current_ids: torch.Tensor,
    prompt_length: int,
    max_new_tokens: int,
    draft_forward_calls: int,
    target_forward_calls: int,
    proposed_tokens: int,
    accepted_draft_tokens: int,
    speculative_rounds: int,
    draft_prefill_time_seconds: float,
    target_prefill_time_seconds: float,
    first_token_time_seconds: float | None,
    total_start: float,
    device: torch.device,
    verification_mode: SpeculativeVerificationMode,
) -> SpeculativeOutput:
    current_ids = current_ids[:, : prompt_length + max_new_tokens]
    generated_token_ids = current_ids[:, prompt_length:]

    _synchronize_if_needed(device)
    total_time_seconds = perf_counter() - total_start
    if first_token_time_seconds is None:
        first_token_time_seconds = total_time_seconds

    return SpeculativeOutput(
        output_ids=current_ids,
        generated_token_ids=generated_token_ids,
        draft_forward_calls=draft_forward_calls,
        target_forward_calls=target_forward_calls,
        proposed_tokens=proposed_tokens,
        accepted_draft_tokens=accepted_draft_tokens,
        speculative_rounds=speculative_rounds,
        draft_prefill_time_seconds=draft_prefill_time_seconds,
        target_prefill_time_seconds=target_prefill_time_seconds,
        time_to_first_token_seconds=first_token_time_seconds,
        total_time_seconds=total_time_seconds,
        verification_mode=verification_mode,
    )


@torch.inference_mode()
def _greedy_speculative_decode_exact(
    draft_model: Any,
    target_model: Any,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    draft_tokens_per_round: int,
    eos_token_id: int | Sequence[int] | None,
) -> SpeculativeOutput:
    """Target-consistent speculative decoding.

    The target verifies candidate tokens with the same one-token incremental
    path used by ``greedy_decode``. For a deterministic target in evaluation
    mode, this guarantees token identity with the target-only greedy reference
    even when low-precision block attention uses a numerically different kernel.
    It is a correctness and draft-acceptance mode,
    not a speed mode: target forward-call count is intentionally comparable to
    target-only decoding.
    """

    eos_token_ids = _normalize_eos_token_ids(eos_token_id)
    device = input_ids.device
    prompt_length = int(input_ids.shape[1])
    current_ids = input_ids.clone()

    _synchronize_if_needed(device)
    total_start = perf_counter()
    prefill = _prefill_pair(draft_model, target_model, current_ids)

    draft_cache = prefill.draft_cache
    draft_next_logits = prefill.draft_next_logits
    target_cache = prefill.target_cache
    target_next_logits = prefill.target_next_logits
    draft_forward_calls = prefill.draft_forward_calls
    target_forward_calls = prefill.target_forward_calls

    proposed_tokens = 0
    accepted_draft_tokens = 0
    speculative_rounds = 0
    first_token_time_seconds: float | None = None

    while current_ids.shape[1] - prompt_length < max_new_tokens:
        speculative_rounds += 1
        generated_count = int(current_ids.shape[1] - prompt_length)
        remaining_tokens = max_new_tokens - generated_count
        proposal_length = min(draft_tokens_per_round, remaining_tokens)

        (
            draft_proposals,
            proposal_cache,
            _,
            round_draft_calls,
        ) = _draft_propose(
            draft_model,
            draft_cache=draft_cache,
            draft_next_logits=draft_next_logits,
            num_tokens=proposal_length,
            eos_token_id=eos_token_id,
        )
        draft_forward_calls += round_draft_calls

        actual_proposal_length = int(draft_proposals.shape[1])
        proposed_tokens += actual_proposal_length
        context_length = int(current_ids.shape[1])
        all_accepted = True
        stop_generation = False

        for proposal_index in range(actual_proposal_length):
            proposal_token = draft_proposals[
                :, proposal_index : proposal_index + 1
            ]
            target_token = torch.argmax(
                target_next_logits,
                dim=-1,
                keepdim=True,
            )

            if torch.equal(proposal_token, target_token):
                current_ids = torch.cat([current_ids, proposal_token], dim=1)
                accepted_draft_tokens += 1

                if first_token_time_seconds is None:
                    _synchronize_if_needed(device)
                    first_token_time_seconds = perf_counter() - total_start

                if _is_eos(proposal_token, eos_token_ids):
                    stop_generation = True
                    break
                if current_ids.shape[1] - prompt_length >= max_new_tokens:
                    stop_generation = True
                    break

                target_cache, target_next_logits = _advance_cache(
                    target_model,
                    token=proposal_token,
                    cache=target_cache,
                )
                target_forward_calls += 1
                continue

            all_accepted = False
            draft_cache = _crop_past_key_values(
                proposal_cache,
                context_length + proposal_index,
            )
            current_ids = torch.cat([current_ids, target_token], dim=1)

            if first_token_time_seconds is None:
                _synchronize_if_needed(device)
                first_token_time_seconds = perf_counter() - total_start

            if _is_eos(target_token, eos_token_ids):
                stop_generation = True
                break
            if current_ids.shape[1] - prompt_length >= max_new_tokens:
                stop_generation = True
                break

            target_cache, target_next_logits = _advance_cache(
                target_model,
                token=target_token,
                cache=target_cache,
            )
            target_forward_calls += 1
            draft_cache, draft_next_logits = _advance_cache(
                draft_model,
                token=target_token,
                cache=draft_cache,
            )
            draft_forward_calls += 1
            break

        if stop_generation:
            break
        if not all_accepted:
            continue

        # _draft_propose intentionally leaves the final proposal outside its
        # cache. The target has consumed every accepted proposal sequentially;
        # align the draft cache before emitting the target bonus token.
        last_proposal = draft_proposals[:, -1:]
        draft_cache, draft_next_logits = _advance_cache(
            draft_model,
            token=last_proposal,
            cache=proposal_cache,
        )
        draft_forward_calls += 1

        bonus_token = torch.argmax(
            target_next_logits,
            dim=-1,
            keepdim=True,
        )
        current_ids = torch.cat([current_ids, bonus_token], dim=1)

        if first_token_time_seconds is None:
            _synchronize_if_needed(device)
            first_token_time_seconds = perf_counter() - total_start

        if _is_eos(bonus_token, eos_token_ids):
            break
        if current_ids.shape[1] - prompt_length >= max_new_tokens:
            break

        target_cache, target_next_logits = _advance_cache(
            target_model,
            token=bonus_token,
            cache=target_cache,
        )
        target_forward_calls += 1
        draft_cache, draft_next_logits = _advance_cache(
            draft_model,
            token=bonus_token,
            cache=draft_cache,
        )
        draft_forward_calls += 1

    return _finalize_output(
        current_ids=current_ids,
        prompt_length=prompt_length,
        max_new_tokens=max_new_tokens,
        draft_forward_calls=draft_forward_calls,
        target_forward_calls=target_forward_calls,
        proposed_tokens=proposed_tokens,
        accepted_draft_tokens=accepted_draft_tokens,
        speculative_rounds=speculative_rounds,
        draft_prefill_time_seconds=prefill.draft_prefill_time_seconds,
        target_prefill_time_seconds=prefill.target_prefill_time_seconds,
        first_token_time_seconds=first_token_time_seconds,
        total_start=total_start,
        device=device,
        verification_mode="exact",
    )


@torch.inference_mode()
def _greedy_speculative_decode_block(
    draft_model: Any,
    target_model: Any,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    draft_tokens_per_round: int,
    eos_token_id: int | Sequence[int] | None,
) -> SpeculativeOutput:
    """Fast block-verification speculative decoding.

    The target validates a candidate block in one forward call, reducing target
    calls when acceptance is high. In finite-precision GPU execution, q_len>1
    verification can use a different kernel and cache arithmetic from q_len=1
    greedy decoding. Therefore exact target-only token identity must be measured;
    use ``verification_mode='exact'`` when identity is a hard requirement.
    """

    eos_token_ids = _normalize_eos_token_ids(eos_token_id)
    device = input_ids.device
    prompt_length = int(input_ids.shape[1])
    current_ids = input_ids.clone()

    _synchronize_if_needed(device)
    total_start = perf_counter()
    prefill = _prefill_pair(draft_model, target_model, current_ids)

    draft_cache = prefill.draft_cache
    draft_next_logits = prefill.draft_next_logits
    target_cache = prefill.target_cache
    target_next_logits = prefill.target_next_logits
    draft_forward_calls = prefill.draft_forward_calls
    target_forward_calls = prefill.target_forward_calls

    proposed_tokens = 0
    accepted_draft_tokens = 0
    speculative_rounds = 0
    first_token_time_seconds: float | None = None

    while current_ids.shape[1] - prompt_length < max_new_tokens:
        speculative_rounds += 1
        generated_count = int(current_ids.shape[1] - prompt_length)
        remaining_tokens = max_new_tokens - generated_count
        proposal_length = min(draft_tokens_per_round, remaining_tokens)

        (
            draft_proposals,
            proposal_cache,
            _,
            round_draft_calls,
        ) = _draft_propose(
            draft_model,
            draft_cache=draft_cache,
            draft_next_logits=draft_next_logits,
            num_tokens=proposal_length,
            eos_token_id=eos_token_id,
        )
        draft_forward_calls += round_draft_calls

        actual_proposal_length = int(draft_proposals.shape[1])
        proposed_tokens += actual_proposal_length
        context_length = int(current_ids.shape[1])

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
                accepted_prefix = draft_proposals[:, :mismatch_index]
                current_ids = torch.cat([current_ids, accepted_prefix], dim=1)
                accepted_draft_tokens += mismatch_index

            accepted_context_length = context_length + mismatch_index
            target_cache = _crop_past_key_values(
                verification_cache,
                accepted_context_length,
            )
            draft_cache = _crop_past_key_values(
                proposal_cache,
                accepted_context_length,
            )

            correction_token = target_predictions[
                :, mismatch_index : mismatch_index + 1
            ]
            current_ids = torch.cat([current_ids, correction_token], dim=1)

            if first_token_time_seconds is None:
                _synchronize_if_needed(device)
                first_token_time_seconds = perf_counter() - total_start

            if _is_eos(correction_token, eos_token_ids):
                break
            if current_ids.shape[1] - prompt_length >= max_new_tokens:
                break

            target_cache, target_next_logits = _advance_cache(
                target_model,
                token=correction_token,
                cache=target_cache,
            )
            target_forward_calls += 1
            draft_cache, draft_next_logits = _advance_cache(
                draft_model,
                token=correction_token,
                cache=draft_cache,
            )
            draft_forward_calls += 1

        else:
            current_ids = torch.cat([current_ids, draft_proposals], dim=1)
            accepted_draft_tokens += actual_proposal_length
            target_cache = verification_cache

            if first_token_time_seconds is None:
                _synchronize_if_needed(device)
                first_token_time_seconds = perf_counter() - total_start

            last_proposal = draft_proposals[:, -1:]
            if _is_eos(last_proposal, eos_token_ids):
                break
            if current_ids.shape[1] - prompt_length >= max_new_tokens:
                break

            # The proposal helper deliberately leaves the final proposal out of
            # the cache. Consume it now that the target accepted the full block.
            draft_cache, draft_next_logits = _advance_cache(
                draft_model,
                token=last_proposal,
                cache=proposal_cache,
            )
            draft_forward_calls += 1

            bonus_token = torch.argmax(
                target_outputs.logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )
            current_ids = torch.cat([current_ids, bonus_token], dim=1)

            if _is_eos(bonus_token, eos_token_ids):
                break
            if current_ids.shape[1] - prompt_length >= max_new_tokens:
                break

            target_cache, target_next_logits = _advance_cache(
                target_model,
                token=bonus_token,
                cache=target_cache,
            )
            target_forward_calls += 1
            draft_cache, draft_next_logits = _advance_cache(
                draft_model,
                token=bonus_token,
                cache=draft_cache,
            )
            draft_forward_calls += 1

    return _finalize_output(
        current_ids=current_ids,
        prompt_length=prompt_length,
        max_new_tokens=max_new_tokens,
        draft_forward_calls=draft_forward_calls,
        target_forward_calls=target_forward_calls,
        proposed_tokens=proposed_tokens,
        accepted_draft_tokens=accepted_draft_tokens,
        speculative_rounds=speculative_rounds,
        draft_prefill_time_seconds=prefill.draft_prefill_time_seconds,
        target_prefill_time_seconds=prefill.target_prefill_time_seconds,
        first_token_time_seconds=first_token_time_seconds,
        total_start=total_start,
        device=device,
        verification_mode="block",
    )


@torch.inference_mode()
def greedy_speculative_decode(
    draft_model: Any,
    target_model: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    draft_tokens_per_round: int = 4,
    eos_token_id: int | Sequence[int] | None = None,
    verification_mode: str = "exact",
) -> SpeculativeOutput:
    """Greedy speculative decoding with persistent draft and target caches.

    ``verification_mode='exact'`` uses q_len=1 target verification and is
    token-identical to target-only greedy decoding for a deterministic target
    in evaluation mode. ``'block'`` performs the
    conventional one-call block verification used for speed experiments; its
    exact-match rate must be reported because low-precision GPU kernels may not
    be bitwise prefix-equivalent across query lengths.
    """

    _validate_inputs(
        input_ids,
        max_new_tokens=max_new_tokens,
        draft_tokens_per_round=draft_tokens_per_round,
    )
    mode = _normalize_verification_mode(verification_mode)

    if mode == "exact":
        return _greedy_speculative_decode_exact(
            draft_model,
            target_model,
            input_ids,
            max_new_tokens=max_new_tokens,
            draft_tokens_per_round=draft_tokens_per_round,
            eos_token_id=eos_token_id,
        )

    return _greedy_speculative_decode_block(
        draft_model,
        target_model,
        input_ids,
        max_new_tokens=max_new_tokens,
        draft_tokens_per_round=draft_tokens_per_round,
        eos_token_id=eos_token_id,
    )
