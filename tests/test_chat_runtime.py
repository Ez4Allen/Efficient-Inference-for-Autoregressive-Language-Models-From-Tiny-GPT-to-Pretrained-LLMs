from __future__ import annotations

from dataclasses import dataclass

import torch

from src.inference.chat_runtime import QwenPairRuntime
from src.models.loader import ModelBundle, SpeculativeModelBundle
from src.models.runtime_config import (
    GenerationConfig,
    ModelEndpointConfig,
    QwenPairConfig,
)


@dataclass
class ToyCache:
    tokens: torch.Tensor

    def crop(self, sequence_length: int) -> None:
        self.tokens = self.tokens[:, :sequence_length]


@dataclass
class ToyOutput:
    logits: torch.Tensor
    past_key_values: ToyCache


class ToyModel:
    def __init__(self, offset: int = 1) -> None:
        self.offset = offset
        self.config = type("Config", (), {"max_position_embeddings": 128})()

    def __call__(self, *, input_ids, past_key_values=None, use_cache=True):
        del use_cache
        prefix = past_key_values.tokens if past_key_values is not None else input_ids[:, :0]
        full = torch.cat([prefix, input_ids], dim=1)
        vocab_size = 32
        predictions = (input_ids + self.offset) % vocab_size
        logits = torch.full((*input_ids.shape, vocab_size), -1000.0)
        logits.scatter_(2, predictions.unsqueeze(-1), 1000.0)
        return ToyOutput(logits, ToyCache(full))


class ToyTokenizer:
    eos_token_id = None
    bos_token_id = 1
    pad_token_id = 0
    unk_token_id = 3

    def apply_chat_template(self, conversation, **kwargs):
        del conversation, kwargs
        return torch.tensor([[2, 4]], dtype=torch.long)

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(int(value)) for value in ids)

    def get_vocab(self):
        return {str(index): index for index in range(32)}

    def get_added_vocab(self):
        return {}

    def __len__(self):
        return 32


def bundle(name: str, model: ToyModel) -> ModelBundle:
    return ModelBundle(
        model_name=name,
        model=model,
        tokenizer=ToyTokenizer(),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def runtime() -> QwenPairRuntime:
    endpoint = ModelEndpointConfig("toy")
    config = QwenPairConfig(
        draft=endpoint,
        target=endpoint,
        device="cpu",
        dtype="fp32",
        generation=GenerationConfig(max_new_tokens=5),
    )
    result = QwenPairRuntime(config)
    result._draft = bundle("draft", ToyModel(offset=1))
    result._target = bundle("target", ToyModel(offset=1))
    result._pair = SpeculativeModelBundle(result._draft, result._target)
    return result


def test_runtime_target_and_speculative_outputs_match() -> None:
    pair_runtime = runtime()
    messages = [{"role": "user", "content": "hello"}]

    target = pair_runtime.generate(messages, engine="target", max_new_tokens=5)
    speculative = pair_runtime.generate(
        messages,
        engine="speculative",
        max_new_tokens=5,
        draft_tokens_per_round=3,
    )

    assert target.text == speculative.text
    assert target.generated_tokens == 5
    assert speculative.acceptance_rate == 1.0
    assert speculative.draft_forward_calls > 0
