
from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.modeling_utils import PreTrainedModel


@dataclass
class AutoregressiveOutput:
    """
    Output produced by autoregressive decoding.
    """

    output_ids: torch.Tensor
    generated_token_ids: torch.Tensor
    target_forward_calls: int


@torch.inference_mode()
def greedy_decode(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> AutoregressiveOutput:
    """
    Generate tokens using standard greedy autoregressive decoding.

    The first forward pass processes the complete prompt and creates
    the KV cache. Each later forward pass processes only the newest token.

    Args:
        model:
            Hugging Face causal language model.

        input_ids:
            Prompt token IDs with shape:
            [batch_size, prompt_length]

        max_new_tokens:
            Maximum number of new tokens to generate.

        eos_token_id:
            Optional end-of-sequence token ID.

    Returns:
        AutoregressiveOutput containing the complete sequence,
        generated token IDs, and target forward-call count.
    """

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than zero.")

    if input_ids.ndim != 2:
        raise ValueError(
            "input_ids must have shape [batch_size, sequence_length]."
        )

    generated_ids = input_ids
    new_token_ids: list[torch.Tensor] = []

    past_key_values = None
    target_forward_calls = 0

    for _ in range(max_new_tokens):
        if past_key_values is None:
            # Prefill: process the complete prompt.
            model_input_ids = generated_ids
        else:
            # Decode: only process the newest token.
            model_input_ids = generated_ids[:, -1:]

        outputs = model(
            input_ids=model_input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )

        target_forward_calls += 1

        next_token_logits = outputs.logits[:, -1, :]

        next_token_id = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True,
        )

        generated_ids = torch.cat(
            [generated_ids, next_token_id],
            dim=-1,
        )

        new_token_ids.append(next_token_id)

        past_key_values = outputs.past_key_values

        if eos_token_id is not None:
            if torch.all(next_token_id == eos_token_id):
                break

    generated_token_ids = torch.cat(
        new_token_ids,
        dim=-1,
    )

    return AutoregressiveOutput(
        output_ids=generated_ids,
        generated_token_ids=generated_token_ids,
        target_forward_calls=target_forward_calls,
    )
