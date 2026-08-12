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


def _filter_sampling_logits(
    logits: torch.Tensor,
    *,
    top_k: int | None,
    top_p: float | None,
) -> torch.Tensor:
    filtered = logits
    if top_k is not None:
        top_k = max(1, min(int(top_k), filtered.shape[-1]))
        threshold = torch.topk(filtered, k=top_k, dim=-1).values[..., -1, None]
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))
    if top_p is not None:
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1].")
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative > float(top_p)
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        restored = torch.full_like(filtered, float("-inf"))
        filtered = restored.scatter(-1, sorted_indices, sorted_logits)
    return filtered


@torch.inference_mode()
def sample_decode(
    model: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    eos_token_id: int | Sequence[int] | None = None,
    temperature: float = 0.8,
    top_p: float | None = 0.9,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> AutoregressiveOutput:
    """Sample autoregressively with persistent KV cache.

    This is used only for custom-model diversity diagnostics.  The production
    GameGuideLM path and speculative correctness tests remain greedy.
    """

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than zero.")
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence].")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    eos_token_ids = _normalize_eos_token_ids(eos_token_id)
    generated_ids = input_ids
    new_token_ids: list[torch.Tensor] = []
    past_key_values = None
    forward_calls = 0
    prefill_time_seconds = 0.0
    decode_times_seconds: list[float] = []
    device = input_ids.device
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=device)

    _synchronize_if_needed(device)
    total_start = perf_counter()

    for step in range(max_new_tokens):
        model_input_ids = generated_ids if past_key_values is None else generated_ids[:, -1:]
        _synchronize_if_needed(device)
        forward_start = perf_counter()
        outputs = model(
            input_ids=model_input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        _synchronize_if_needed(device)
        elapsed = perf_counter() - forward_start
        if step == 0:
            prefill_time_seconds = elapsed
        else:
            decode_times_seconds.append(elapsed)
        forward_calls += 1

        logits = outputs.logits[:, -1, :].float() / float(temperature)
        logits = _filter_sampling_logits(logits, top_k=top_k, top_p=top_p)
        probabilities = torch.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(
            probabilities,
            num_samples=1,
            generator=generator,
        )
        if eos_token_ids:
            eos_fill = torch.full_like(next_token_id, eos_token_ids[0])
            next_token_id = torch.where(finished.unsqueeze(-1), eos_fill, next_token_id)

        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
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
    generated_token_ids = torch.cat(new_token_ids, dim=-1)
    return AutoregressiveOutput(
        output_ids=generated_ids,
        generated_token_ids=generated_token_ids,
        target_forward_calls=forward_calls,
        prefill_time_seconds=prefill_time_seconds,
        decode_times_seconds=decode_times_seconds,
        total_time_seconds=total_time_seconds,
    )
