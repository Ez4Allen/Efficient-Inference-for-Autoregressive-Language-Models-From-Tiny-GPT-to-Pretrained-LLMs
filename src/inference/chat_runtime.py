"""Text-chat runtime shared by grounded generation and inference experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, Mapping, Sequence

import torch

from src.inference.autoregressive import AutoregressiveOutput, greedy_decode
from src.inference.speculative import SpeculativeOutput, greedy_speculative_decode
from src.models.loader import (
    ModelBundle,
    SpeculativeModelBundle,
    load_causal_lm,
    resolve_dtype_name,
    validate_tokenizer_compatibility,
)
from src.models.runtime_config import ModelEndpointConfig, QwenPairConfig


@dataclass(slots=True)
class ChatGenerationResult:
    text: str
    engine: str
    prompt_tokens: int
    generated_tokens: int
    total_time_seconds: float
    ttft_seconds: float
    mean_tpot_seconds: float
    tokens_per_second: float
    target_forward_calls: int
    draft_forward_calls: int = 0
    proposed_tokens: int = 0
    accepted_draft_tokens: int = 0
    speculative_rounds: int = 0
    acceptance_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _endpoint_kwargs(endpoint: ModelEndpointConfig) -> dict[str, Any]:
    return {
        "model_name": endpoint.model_name_or_path,
        "adapter_path": endpoint.adapter_path,
        "trust_remote_code": endpoint.trust_remote_code,
        "local_files_only": endpoint.local_files_only,
        "load_in_4bit": endpoint.load_in_4bit,
    }


def _apply_chat_template(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    enable_thinking: bool,
) -> torch.Tensor:
    common = {
        "conversation": list(messages),
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
    }
    try:
        encoded = tokenizer.apply_chat_template(
            **common,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        encoded = tokenizer.apply_chat_template(**common)

    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    if not isinstance(encoded, torch.Tensor):
        encoded = torch.tensor(encoded, dtype=torch.long)
    if encoded.ndim == 1:
        encoded = encoded.unsqueeze(0)
    return encoded




def _eos_token_id(bundle: ModelBundle) -> int | list[int] | None:
    generation_config = getattr(bundle.model, "generation_config", None)
    value = getattr(generation_config, "eos_token_id", None)
    if value is not None:
        return value
    return bundle.tokenizer.eos_token_id


def _context_limit(bundle: ModelBundle) -> int | None:
    config = getattr(bundle.model, "config", None)
    for attribute in ("max_position_embeddings", "n_positions", "max_sequence_length"):
        value = getattr(config, attribute, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _validate_context(bundle: ModelBundle, prompt_tokens: int, max_new_tokens: int) -> None:
    limit = _context_limit(bundle)
    if limit is not None and prompt_tokens + max_new_tokens > limit:
        raise ValueError(
            "Prompt and requested output exceed the model context limit: "
            f"{prompt_tokens} + {max_new_tokens} > {limit}."
        )


def _autoregressive_result(
    output: AutoregressiveOutput,
    *,
    tokenizer: Any,
    engine: str,
    prompt_tokens: int,
) -> ChatGenerationResult:
    generated_tokens = int(output.generated_token_ids.shape[1])
    text = tokenizer.decode(
        output.generated_token_ids[0],
        skip_special_tokens=True,
    ).strip()
    mean_tpot = mean(output.decode_times_seconds) if output.decode_times_seconds else 0.0
    throughput = generated_tokens / output.total_time_seconds if output.total_time_seconds > 0 else 0.0
    return ChatGenerationResult(
        text=text,
        engine=engine,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        total_time_seconds=output.total_time_seconds,
        ttft_seconds=output.prefill_time_seconds,
        mean_tpot_seconds=mean_tpot,
        tokens_per_second=throughput,
        target_forward_calls=output.target_forward_calls,
    )


def _speculative_result(
    output: SpeculativeOutput,
    *,
    tokenizer: Any,
    prompt_tokens: int,
) -> ChatGenerationResult:
    generated_tokens = int(output.generated_token_ids.shape[1])
    text = tokenizer.decode(
        output.generated_token_ids[0],
        skip_special_tokens=True,
    ).strip()
    remaining_time = max(0.0, output.total_time_seconds - output.target_prefill_time_seconds)
    mean_tpot = remaining_time / max(1, generated_tokens - 1)
    throughput = generated_tokens / output.total_time_seconds if output.total_time_seconds > 0 else 0.0
    return ChatGenerationResult(
        text=text,
        engine="speculative",
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        total_time_seconds=output.total_time_seconds,
        ttft_seconds=output.target_prefill_time_seconds,
        mean_tpot_seconds=mean_tpot,
        tokens_per_second=throughput,
        target_forward_calls=output.target_forward_calls,
        draft_forward_calls=output.draft_forward_calls,
        proposed_tokens=output.proposed_tokens,
        accepted_draft_tokens=output.accepted_draft_tokens,
        speculative_rounds=output.speculative_rounds,
        acceptance_rate=output.acceptance_rate,
    )


class QwenPairRuntime:
    """Lazy paired-model runtime with target, draft, and baseline speculative modes."""

    def __init__(self, config: QwenPairConfig) -> None:
        self.config = config
        self._draft: ModelBundle | None = None
        self._target: ModelBundle | None = None
        self._pair: SpeculativeModelBundle | None = None

    def _load_endpoint(self, endpoint: ModelEndpointConfig) -> ModelBundle:
        return load_causal_lm(
            **_endpoint_kwargs(endpoint),
            device=self.config.device,
            dtype=resolve_dtype_name(self.config.dtype),
        )

    def load_target(self) -> ModelBundle:
        if self._target is None:
            self._target = self._load_endpoint(self.config.target)
        return self._target

    def load_draft(self) -> ModelBundle:
        if self._draft is None:
            self._draft = self._load_endpoint(self.config.draft)
        return self._draft

    def load_pair(self) -> SpeculativeModelBundle:
        if self._pair is None:
            draft = self.load_draft()
            target = self.load_target()
            validate_tokenizer_compatibility(draft, target)
            if draft.device != target.device:
                raise ValueError(
                    "The current speculative decoder requires both models on "
                    f"one device, got {draft.device} and {target.device}."
                )
            self._pair = SpeculativeModelBundle(draft=draft, target=target)
        return self._pair

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        engine: str | None = None,
        max_new_tokens: int | None = None,
        draft_tokens_per_round: int | None = None,
        enable_thinking: bool | None = None,
    ) -> ChatGenerationResult:
        selected_engine = str(engine or self.config.generation.engine).casefold()
        max_tokens = int(max_new_tokens or self.config.generation.max_new_tokens)
        draft_tokens = int(
            draft_tokens_per_round or self.config.generation.draft_tokens_per_round
        )
        thinking = (
            self.config.generation.enable_thinking
            if enable_thinking is None
            else bool(enable_thinking)
        )

        if selected_engine == "target":
            bundle = self.load_target()
            input_ids = _apply_chat_template(
                bundle.tokenizer,
                messages,
                enable_thinking=thinking,
            ).to(bundle.device)
            _validate_context(bundle, input_ids.shape[1], max_tokens)
            output = greedy_decode(
                bundle.model,
                input_ids,
                max_new_tokens=max_tokens,
                eos_token_id=_eos_token_id(bundle),
            )
            return _autoregressive_result(
                output,
                tokenizer=bundle.tokenizer,
                engine="target",
                prompt_tokens=int(input_ids.shape[1]),
            )

        if selected_engine == "draft":
            bundle = self.load_draft()
            input_ids = _apply_chat_template(
                bundle.tokenizer,
                messages,
                enable_thinking=thinking,
            ).to(bundle.device)
            _validate_context(bundle, input_ids.shape[1], max_tokens)
            output = greedy_decode(
                bundle.model,
                input_ids,
                max_new_tokens=max_tokens,
                eos_token_id=_eos_token_id(bundle),
            )
            return _autoregressive_result(
                output,
                tokenizer=bundle.tokenizer,
                engine="draft",
                prompt_tokens=int(input_ids.shape[1]),
            )

        if selected_engine == "speculative":
            pair = self.load_pair()
            input_ids = _apply_chat_template(
                pair.target.tokenizer,
                messages,
                enable_thinking=thinking,
            ).to(pair.target.device)
            _validate_context(pair.target, input_ids.shape[1], max_tokens)
            output = greedy_speculative_decode(
                pair.draft.model,
                pair.target.model,
                input_ids,
                max_new_tokens=max_tokens,
                draft_tokens_per_round=draft_tokens,
                eos_token_id=_eos_token_id(pair.target),
            )
            return _speculative_result(
                output,
                tokenizer=pair.target.tokenizer,
                prompt_tokens=int(input_ids.shape[1]),
            )

        raise ValueError("engine must be target, draft, or speculative.")

    def close(self) -> None:
        self._pair = None
        self._draft = None
        self._target = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
