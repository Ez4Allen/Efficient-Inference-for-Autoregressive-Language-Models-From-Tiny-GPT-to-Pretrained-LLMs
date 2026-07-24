from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from collections.abc import Sequence
from typing import Any

import torch


@dataclass
class AutoregressiveOutput:
    output_ids: torch.Tensor
    generated_token_ids: torch.Tensor
    target_forward_calls: int
    prefill_time_seconds: float
    decode_times_seconds: list[float]
    total_time_seconds: float


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


@torch.inference_mode()
def greedy_decode(
    model: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | Sequence[int] | None = None,
) -> AutoregressiveOutput:
    """
    Generate tokens using standard greedy autoregressive decoding.

    The first forward pass processes the complete prompt and creates
    the KV cache. Each later forward pass processes only the newest token.
    """

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than zero.")

    if input_ids.ndim != 2:
        raise ValueError(
            "input_ids must have shape [batch_size, sequence_length]."
        )

    eos_token_ids = _normalize_eos_token_ids(eos_token_id)
    generated_ids = input_ids
    new_token_ids: list[torch.Tensor] = []

    past_key_values = None
    target_forward_calls = 0

    prefill_time_seconds = 0.0
    decode_times_seconds: list[float] = []

    device = input_ids.device
    finished = torch.zeros(
        input_ids.shape[0],
        dtype=torch.bool,
        device=device,
    )

    _synchronize_if_needed(device)
    total_start = perf_counter()

    for step in range(max_new_tokens):
        if past_key_values is None:
            model_input_ids = generated_ids
        else:
            model_input_ids = generated_ids[:, -1:]

        _synchronize_if_needed(device)
        forward_start = perf_counter()

        outputs = model(
            input_ids=model_input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )

        _synchronize_if_needed(device)
        forward_elapsed = perf_counter() - forward_start

        if step == 0:
            prefill_time_seconds = forward_elapsed
        else:
            decode_times_seconds.append(forward_elapsed)

        target_forward_calls += 1

        next_token_logits = outputs.logits[:, -1, :]

        next_token_id = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True,
        )

        if eos_token_ids:
            eos_fill = torch.full_like(next_token_id, eos_token_ids[0])
            next_token_id = torch.where(
                finished.unsqueeze(-1),
                eos_fill,
                next_token_id,
            )

        generated_ids = torch.cat(
            [generated_ids, next_token_id],
            dim=-1,
        )

        new_token_ids.append(next_token_id)

        past_key_values = outputs.past_key_values

        if eos_token_ids:
            eos_tensor = torch.tensor(
                eos_token_ids,
                dtype=next_token_id.dtype,
                device=next_token_id.device,
            )
            finished |= torch.isin(next_token_id.squeeze(-1), eos_tensor)
            if bool(torch.all(finished)):
                break

    _synchronize_if_needed(device)
    total_time_seconds = perf_counter() - total_start

    generated_token_ids = torch.cat(
        new_token_ids,
        dim=-1,
    )

    return AutoregressiveOutput(
        output_ids=generated_ids,
        generated_token_ids=generated_token_ids,
        target_forward_calls=target_forward_calls,
        prefill_time_seconds=prefill_time_seconds,
        decode_times_seconds=decode_times_seconds,
        total_time_seconds=total_time_seconds,
    )
