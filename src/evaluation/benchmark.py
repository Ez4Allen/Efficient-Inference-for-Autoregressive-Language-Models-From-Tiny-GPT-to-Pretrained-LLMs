"""Reusable benchmark-case orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch

from src.data.prompt_builder import build_prompt_batch, supported_prompt_types
from src.evaluation.gpu_monitor import get_gpu_info
from src.evaluation.prefill_decode import benchmark_prefill_decode
from src.models.loader import ModelBundle, get_parameter_count
from src.utils.io import read_yaml


@dataclass(frozen=True)
class BenchmarkCase:
    prompt_length: int
    output_length: int
    prompt_type: str = "technical"
    batch_size: int = 1

    def __post_init__(self) -> None:
        if self.prompt_length < 1:
            raise ValueError("prompt_length must be at least 1.")
        if self.output_length < 1:
            raise ValueError("output_length must be at least 1.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if self.prompt_type.casefold() not in supported_prompt_types():
            raise ValueError(
                "Unsupported prompt_type. Expected one of: "
                + ", ".join(supported_prompt_types())
            )

    @property
    def key(self) -> tuple[int, int, str, int]:
        return (
            self.prompt_length,
            self.output_length,
            self.prompt_type.casefold(),
            self.batch_size,
        )


def resolve_torch_dtype(value: str | torch.dtype | None) -> torch.dtype | None:
    """Convert a configuration dtype into a PyTorch dtype.

    ``None`` and ``auto`` return ``None`` so the model loader can choose a sensible
    device-specific default.
    """

    if value is None or isinstance(value, torch.dtype):
        return value
    normalized = str(value).strip().casefold().replace("torch.", "")
    if normalized in {"", "auto", "none"}:
        return None
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(
            "Unsupported dtype. Expected auto, float32, float16, or bfloat16."
        )
    return mapping[normalized]


def load_benchmark_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a benchmark YAML configuration."""

    value = read_yaml(path)
    if not isinstance(value, dict):
        raise TypeError("Benchmark configuration must be a YAML mapping.")

    required = {
        "model_name",
        "prompt_lengths",
        "output_lengths",
        "prompt_types",
        "batch_sizes",
        "runs",
        "warmup_runs",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError("Benchmark configuration is missing: " + ", ".join(missing))
    return value


def iter_benchmark_cases(config: Mapping[str, Any]) -> Iterator[BenchmarkCase]:
    """Yield the Cartesian benchmark sweep in deterministic order."""

    for prompt_length, output_length, prompt_type, batch_size in product(
        config["prompt_lengths"],
        config["output_lengths"],
        config["prompt_types"],
        config["batch_sizes"],
    ):
        yield BenchmarkCase(
            prompt_length=int(prompt_length),
            output_length=int(output_length),
            prompt_type=str(prompt_type),
            batch_size=int(batch_size),
        )


def model_context_length(bundle: ModelBundle) -> int | None:
    """Return a known context-window limit from common model config fields."""

    config = getattr(bundle.model, "config", None)
    if config is None:
        return None
    for attribute in (
        "max_position_embeddings",
        "n_positions",
        "max_sequence_length",
        "seq_length",
    ):
        value = getattr(config, attribute, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def run_benchmark_case(
    bundle: ModelBundle,
    case: BenchmarkCase,
    *,
    warmup_runs: int,
    measured_runs: int,
    enforce_context_limit: bool = True,
) -> dict[str, Any]:
    """Run one exact token-shape case and return a JSON-serializable record."""

    context_length = model_context_length(bundle)
    requested_total_length = case.prompt_length + case.output_length
    if (
        enforce_context_limit
        and context_length is not None
        and requested_total_length > context_length
    ):
        raise ValueError(
            f"Case requires {requested_total_length} tokens but model context length "
            f"is {context_length}."
        )

    prompt_batch = build_prompt_batch(
        bundle.tokenizer,
        case.prompt_length,
        prompt_type=case.prompt_type,
        batch_size=case.batch_size,
        device=bundle.device,
    )
    benchmark = benchmark_prefill_decode(
        bundle.model,
        prompt_batch.input_ids,
        max_new_tokens=case.output_length,
        eos_token_id=None,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
    )

    return {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": bundle.model_name,
            "class": bundle.model.__class__.__name__,
            "device": str(bundle.device),
            "dtype": str(bundle.dtype).replace("torch.", ""),
            "parameter_count": get_parameter_count(bundle.model),
            "context_length": context_length,
        },
        "case": asdict(case),
        "prompt": {
            "actual_tokens": prompt_batch.actual_tokens,
            "text_preview": prompt_batch.text[:240],
        },
        "benchmark": benchmark.to_dict(),
        "accelerator": get_gpu_info(bundle.device),
    }
