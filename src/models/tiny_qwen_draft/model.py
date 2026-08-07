"""A small Qwen-style causal LM designed for speculative drafting.

The implementation is intentionally self-contained and uses only PyTorch.  It
shares the target tokenizer, exposes a Hugging-Face-like causal-LM forward
interface, and supports persistent KV caching so it can be consumed by the
project's speculative decoder.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cache import LayerCache, TinyQwenDraftCache
from .config import TinyQwenDraftConfig


@dataclass
class TinyCausalLMOutputWithPast:
    """Minimal causal-LM output compatible with project inference code."""

    logits: torch.Tensor
    past_key_values: TinyQwenDraftCache | None = None
    loss: torch.Tensor | None = None


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states.float() * torch.rsqrt(variance + self.eps)
        return self.weight * normalized.to(input_dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        inverse_frequency = 1.0 / (
            float(theta)
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32)
                / float(head_dim)
            )
        )
        self.register_buffer("inv_freq", inverse_frequency, persistent=False)

    def forward(
        self,
        position_ids: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frequencies = torch.einsum(
            "bt,d->btd",
            position_ids.float(),
            self.inv_freq.float(),
        )
        embedding = torch.cat([frequencies, frequencies], dim=-1)
        cosine = embedding.cos().to(dtype=dtype).unsqueeze(1)
        sine = embedding.sin().to(dtype=dtype).unsqueeze(1)
        return cosine, sine


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rotary(
    query: torch.Tensor,
    key: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    query = (query * cosine) + (_rotate_half(query) * sine)
    key = (key * cosine) + (_rotate_half(key) * sine)
    return query, key


def _repeat_kv(hidden_states: torch.Tensor, repetitions: int) -> torch.Tensor:
    if repetitions == 1:
        return hidden_states
    return hidden_states.repeat_interleave(repetitions, dim=1)


def _cache_layers(
    past_key_values: Any,
    expected_layers: int,
) -> list[LayerCache | None]:
    if past_key_values is None:
        return [None] * expected_layers

    if isinstance(past_key_values, TinyQwenDraftCache):
        layers: Sequence[Sequence[torch.Tensor]] = past_key_values.layers
    elif hasattr(past_key_values, "to_legacy_cache"):
        layers = past_key_values.to_legacy_cache()
    elif isinstance(past_key_values, (tuple, list)):
        layers = past_key_values
    else:
        raise TypeError(
            "past_key_values must be TinyQwenDraftCache, a legacy cache, or null."
        )

    if len(layers) != expected_layers:
        raise ValueError(
            f"Expected {expected_layers} cache layers, got {len(layers)}."
        )

    result: list[LayerCache | None] = []
    for index, layer in enumerate(layers):
        if layer is None:
            result.append(None)
            continue
        if len(layer) != 2:
            raise ValueError(f"Cache layer {index} must contain key and value.")
        result.append((layer[0], layer[1]))
    return result


def _past_length(layers: Sequence[LayerCache | None]) -> int:
    lengths = {
        int(layer[0].shape[-2])
        for layer in layers
        if layer is not None
    }
    if not lengths:
        return 0
    if len(lengths) != 1:
        raise ValueError("All KV-cache layers must have the same sequence length.")
    return lengths.pop()


def _attention_bias(
    *,
    batch_size: int,
    query_length: int,
    key_length: int,
    past_length: int,
    attention_mask: torch.Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    query_positions = (
        torch.arange(query_length, device=device).view(query_length, 1)
        + past_length
    )
    key_positions = torch.arange(key_length, device=device).view(1, key_length)
    allowed = key_positions <= query_positions
    allowed = allowed.view(1, 1, query_length, key_length).expand(
        batch_size,
        1,
        query_length,
        key_length,
    )

    if attention_mask is not None:
        if attention_mask.ndim != 2 or attention_mask.shape[0] != batch_size:
            raise ValueError(
                "attention_mask must have shape [batch_size, sequence_length]."
            )
        if attention_mask.shape[1] == query_length and past_length > 0:
            prefix = torch.ones(
                batch_size,
                past_length,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([prefix, attention_mask], dim=1)
        if attention_mask.shape[1] != key_length:
            raise ValueError(
                "attention_mask length must match the total cached sequence: "
                f"{attention_mask.shape[1]} != {key_length}."
            )
        valid_keys = attention_mask.to(device=device, dtype=torch.bool)
        allowed = allowed & valid_keys[:, None, None, :]

    bias = torch.zeros(
        (batch_size, 1, query_length, key_length),
        device=device,
        dtype=dtype,
    )
    return bias.masked_fill(~allowed, torch.finfo(dtype).min)


class TinyQwenAttention(nn.Module):
    def __init__(self, config: TinyQwenDraftConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.dropout = float(config.attention_dropout)

        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.rotary = RotaryEmbedding(config.head_dim, config.rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_value: LayerCache | None,
        use_cache: bool,
    ) -> tuple[torch.Tensor, LayerCache | None]:
        batch_size, query_length, _ = hidden_states.shape

        query = self.q_norm(
            self.q_proj(hidden_states).view(
                batch_size,
                query_length,
                self.num_heads,
                self.head_dim,
            )
        ).transpose(1, 2)
        key = self.k_norm(
            self.k_proj(hidden_states).view(
                batch_size,
                query_length,
                self.num_key_value_heads,
                self.head_dim,
            )
        ).transpose(1, 2)
        value = self.v_proj(hidden_states).view(
            batch_size,
            query_length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)

        cosine, sine = self.rotary(position_ids, dtype=query.dtype)
        query, key = _apply_rotary(query, key, cosine, sine)

        past_length = 0
        if past_key_value is not None:
            past_key, past_value = past_key_value
            past_length = int(past_key.shape[-2])
            if past_key.shape[:2] != key.shape[:2] or past_key.shape[-1] != key.shape[-1]:
                raise ValueError("Past key shape is incompatible with the current layer.")
            key = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)

        present = (key, value) if use_cache else None
        repeated_key = _repeat_kv(key, self.num_key_value_groups)
        repeated_value = _repeat_kv(value, self.num_key_value_groups)

        bias = _attention_bias(
            batch_size=batch_size,
            query_length=query_length,
            key_length=int(repeated_key.shape[-2]),
            past_length=past_length,
            attention_mask=attention_mask,
            device=hidden_states.device,
            dtype=query.dtype,
        )

        attended = F.scaled_dot_product_attention(
            query,
            repeated_key,
            repeated_value,
            attn_mask=bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size,
            query_length,
            self.hidden_size,
        )
        return self.o_proj(attended), present


class TinyQwenMLP(nn.Module):
    def __init__(self, config: TinyQwenDraftConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class TinyQwenDecoderLayer(nn.Module):
    def __init__(self, config: TinyQwenDraftConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = TinyQwenAttention(config)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.mlp = TinyQwenMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_value: LayerCache | None,
        use_cache: bool,
    ) -> tuple[torch.Tensor, LayerCache | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present = self.self_attn(
            hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states, present


class TinyQwenDraft(nn.Module):
    """Qwen-token-compatible draft model built entirely in this repository."""

    config_class = TinyQwenDraftConfig
    base_model_prefix = "model"

    def __init__(self, config: TinyQwenDraftConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [TinyQwenDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.apply(self._initialize_weights)
        if config.tie_word_embeddings:
            self.tie_weights()

        self.generation_config = SimpleNamespace(
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
            pad_token_id=config.pad_token_id,
            use_cache=config.use_cache,
        )

    def _initialize_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )

    def tie_weights(self) -> None:
        self.lm_head.weight = self.embed_tokens.weight

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value
        if self.config.tie_word_embeddings:
            self.tie_weights()

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        self.lm_head = value
        if self.config.tie_word_embeddings:
            self.tie_weights()

    def num_parameters(self, *, trainable_only: bool = False) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if not trainable_only or parameter.requires_grad
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        return_dict: bool = True,
        cache_position: torch.Tensor | None = None,
        loss_only: bool = False,
        **_: Any,
    ) -> TinyCausalLMOutputWithPast | tuple[Any, ...]:
        del cache_position
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids or inputs_embeds.")

        if input_ids is not None:
            if input_ids.ndim != 2:
                raise ValueError("input_ids must have shape [batch, sequence].")
            if input_ids.dtype != torch.long:
                input_ids = input_ids.long()
            hidden_states = self.embed_tokens(input_ids)
        else:
            if inputs_embeds is None or inputs_embeds.ndim != 3:
                raise ValueError(
                    "inputs_embeds must have shape [batch, sequence, hidden_size]."
                )
            hidden_states = inputs_embeds

        batch_size, sequence_length, _ = hidden_states.shape
        if sequence_length < 1:
            raise ValueError("The input sequence cannot be empty.")

        past_layers = _cache_layers(past_key_values, len(self.layers))
        past_length = _past_length(past_layers)
        total_length = past_length + sequence_length
        if total_length > self.config.max_position_embeddings:
            raise ValueError(
                "Input and cache exceed max_position_embeddings: "
                f"{total_length} > {self.config.max_position_embeddings}."
            )

        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                total_length,
                device=hidden_states.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(batch_size, sequence_length)
        else:
            if position_ids.shape != (batch_size, sequence_length):
                raise ValueError(
                    "position_ids must have shape [batch, sequence_length]."
                )
            position_ids = position_ids.to(
                device=hidden_states.device,
                dtype=torch.long,
            )

        use_cache = self.config.use_cache if use_cache is None else bool(use_cache)
        present_layers: list[LayerCache] = []
        for layer, past_layer in zip(self.layers, past_layers, strict=True):
            hidden_states, present = layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_value=past_layer,
                use_cache=use_cache,
            )
            if present is not None:
                present_layers.append(present)

        hidden_states = self.norm(hidden_states)

        loss = None
        if labels is not None:
            if labels.shape != (batch_size, sequence_length):
                raise ValueError("labels must have the same shape as input_ids.")
            if sequence_length < 2:
                raise ValueError("At least two tokens are required to compute LM loss.")

        if labels is not None and loss_only:
            # SFT masks prompt tokens with -100. Project only hidden states that
            # predict supervised assistant tokens, avoiding a [B, T, 150k+]
            # logits tensor during custom-draft training.
            shifted_labels = labels[:, 1:].to(hidden_states.device)
            supervised = shifted_labels != -100
            if not bool(supervised.any()):
                raise ValueError("No supervised labels are available for loss.")
            selected_hidden = hidden_states[:, :-1, :][supervised]
            selected_labels = shifted_labels[supervised]
            logits = self.lm_head(selected_hidden)
            loss = F.cross_entropy(logits, selected_labels)
        else:
            logits = self.lm_head(hidden_states)
            if labels is not None:
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].to(logits.device).contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

        cache = TinyQwenDraftCache(present_layers) if use_cache else None
        output = TinyCausalLMOutputWithPast(
            logits=logits,
            past_key_values=cache,
            loss=loss,
        )
        if return_dict:
            return output
        if loss is None:
            return logits, cache
        return loss, logits, cache

    def save_pretrained(
        self,
        directory: str | Path,
        *,
        tokenizer: Any | None = None,
    ) -> Path:
        directory = Path(directory).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(directory)

        temporary = directory / "pytorch_model.bin.tmp"
        final = directory / "pytorch_model.bin"
        torch.save(self.state_dict(), temporary)
        temporary.replace(final)

        if tokenizer is not None:
            save = getattr(tokenizer, "save_pretrained", None)
            if not callable(save):
                raise TypeError("tokenizer must implement save_pretrained().")
            save(str(directory))
        return directory

    @classmethod
    def from_pretrained(
        cls,
        directory: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        strict: bool = True,
    ) -> "TinyQwenDraft":
        directory = Path(directory).expanduser().resolve()
        config = TinyQwenDraftConfig.from_pretrained(directory)
        weights_path = directory / "pytorch_model.bin"
        if not weights_path.exists():
            raise FileNotFoundError(f"TinyQwenDraft weights not found: {weights_path}")

        load_kwargs: dict[str, Any] = {"map_location": map_location}
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kwargs["weights_only"] = True
        state_dict = torch.load(weights_path, **load_kwargs)
        if not isinstance(state_dict, dict):
            raise TypeError("TinyQwenDraft checkpoint must contain a state dictionary.")

        model = cls(config)
        model.load_state_dict(state_dict, strict=strict)
        if dtype is not None:
            model.to(dtype=dtype)
        return model
