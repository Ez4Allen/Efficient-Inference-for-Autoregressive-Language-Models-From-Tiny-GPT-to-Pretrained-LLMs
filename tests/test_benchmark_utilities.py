from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from src.data.prompt_builder import build_prompt_batch, make_prompt
from src.evaluation.benchmark import BenchmarkCase, iter_benchmark_cases, resolve_torch_dtype
from src.evaluation.gpu_monitor import get_gpu_info
from src.evaluation.prefill_decode import benchmark_prefill_decode, measure_prefill_decode
from src.utils.io import read_json, read_jsonl, write_json, write_jsonl


class FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def __call__(self, text: str, **kwargs):
        del kwargs
        return {"input_ids": [3 + (ord(character) % 29) for character in text]}

    def decode(self, ids, **kwargs):
        del kwargs
        return " ".join(str(value) for value in ids)


@dataclass
class ToyCache:
    tokens: torch.Tensor


@dataclass
class ToyOutput:
    logits: torch.Tensor
    past_key_values: ToyCache


class ToyCausalLM:
    def __init__(self, vocab_size: int = 64) -> None:
        self.vocab_size = vocab_size

    def __call__(self, *, input_ids, past_key_values=None, use_cache=True):
        del use_cache
        prefix = (
            past_key_values.tokens
            if past_key_values is not None
            else input_ids[:, :0]
        )
        full = torch.cat([prefix, input_ids], dim=1)
        predicted = (input_ids + 1) % self.vocab_size
        logits = torch.full(
            (*input_ids.shape, self.vocab_size),
            -1_000.0,
            device=input_ids.device,
        )
        logits.scatter_(2, predicted.unsqueeze(-1), 1_000.0)
        return ToyOutput(logits=logits, past_key_values=ToyCache(full))


def test_exact_prompt_batch_and_approximate_prompt() -> None:
    tokenizer = FakeTokenizer()
    batch = build_prompt_batch(
        tokenizer,
        37,
        prompt_type="coding",
        batch_size=3,
    )
    assert batch.input_ids.shape == (3, 37)
    assert batch.attention_mask.shape == (3, 37)
    assert torch.all(batch.attention_mask == 1)
    assert batch.actual_tokens == 37
    assert len(make_prompt(12, "qa").split()) == 12


def test_prefill_decode_measurement_counts_batch_tokens() -> None:
    model = ToyCausalLM()
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    run = measure_prefill_decode(model, input_ids, max_new_tokens=4)
    assert run.batch_size == 2
    assert run.generated_tokens_per_sequence == 4
    assert run.total_generated_tokens == 8
    assert run.target_forward_calls == 4
    assert run.throughput_tokens_per_second > 0

    summary = benchmark_prefill_decode(
        model,
        input_ids,
        max_new_tokens=3,
        warmup_runs=1,
        measured_runs=2,
    )
    assert summary.measured_runs == 2
    assert len(summary.runs) == 2
    assert summary.ttft_seconds.mean >= 0


def test_benchmark_case_config_and_dtype_resolution() -> None:
    config = {
        "prompt_lengths": [8, 16],
        "output_lengths": [4],
        "prompt_types": ["technical"],
        "batch_sizes": [1, 2],
    }
    cases = list(iter_benchmark_cases(config))
    assert len(cases) == 4
    assert cases[0] == BenchmarkCase(8, 4, "technical", 1)
    assert resolve_torch_dtype("bf16") is torch.bfloat16
    assert resolve_torch_dtype("auto") is None
    with pytest.raises(ValueError):
        resolve_torch_dtype("int8")


def test_gpu_info_cpu_fallback() -> None:
    info = get_gpu_info("cpu")
    assert info["available"] is False
    assert info["device_type"] == "cpu"
    assert info["memory"]["peak_allocated_bytes"] == 0


def test_json_and_jsonl_io_round_trip(tmp_path: Path) -> None:
    json_path = write_json(tmp_path / "document.json", {"name": "夜之刃"})
    assert read_json(json_path) == {"name": "夜之刃"}

    records = [{"id": 1}, {"id": 2}]
    jsonl_path = write_jsonl(tmp_path / "records.jsonl", records)
    assert read_jsonl(jsonl_path) == records
