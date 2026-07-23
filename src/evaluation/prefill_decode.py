"""Prefill/decode measurement utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean, median, pstdev
from typing import Any

import torch

from src.evaluation.gpu_monitor import begin_memory_measurement, end_memory_measurement
from src.inference.autoregressive import AutoregressiveOutput, greedy_decode


@dataclass(frozen=True)
class MetricDistribution:
    mean: float
    p50: float
    p90: float
    minimum: float
    maximum: float
    std: float


@dataclass
class PrefillDecodeRun:
    """One measured generation run."""

    batch_size: int
    prompt_tokens_per_sequence: int
    generated_tokens_per_sequence: int
    total_generated_tokens: int
    target_forward_calls: int
    ttft_seconds: float
    mean_tpot_seconds: float
    decode_total_seconds: float
    total_latency_seconds: float
    throughput_tokens_per_second: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    raw_output: AutoregressiveOutput

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("raw_output", None)
        return value


@dataclass
class PrefillDecodeBenchmark:
    """Repeated prefill/decode measurements and aggregate distributions."""

    warmup_runs: int
    measured_runs: int
    batch_size: int
    prompt_tokens_per_sequence: int
    requested_new_tokens: int
    runs: list[PrefillDecodeRun]
    ttft_seconds: MetricDistribution
    mean_tpot_seconds: MetricDistribution
    total_latency_seconds: MetricDistribution
    throughput_tokens_per_second: MetricDistribution
    peak_allocated_bytes: MetricDistribution

    def to_dict(self) -> dict[str, Any]:
        return {
            "warmup_runs": self.warmup_runs,
            "measured_runs": self.measured_runs,
            "batch_size": self.batch_size,
            "prompt_tokens_per_sequence": self.prompt_tokens_per_sequence,
            "requested_new_tokens": self.requested_new_tokens,
            "runs": [run.to_dict() for run in self.runs],
            "metrics": {
                "ttft_seconds": asdict(self.ttft_seconds),
                "mean_tpot_seconds": asdict(self.mean_tpot_seconds),
                "total_latency_seconds": asdict(self.total_latency_seconds),
                "throughput_tokens_per_second": asdict(
                    self.throughput_tokens_per_second
                ),
                "peak_allocated_bytes": asdict(self.peak_allocated_bytes),
            },
        }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile from an empty sequence.")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be between 0 and 100.")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] + fraction * (ordered[upper] - ordered[lower]))


def _distribution(values: list[float]) -> MetricDistribution:
    if not values:
        raise ValueError("Cannot summarize an empty sequence.")
    numeric = [float(value) for value in values]
    return MetricDistribution(
        mean=float(mean(numeric)),
        p50=float(median(numeric)),
        p90=_percentile(numeric, 90.0),
        minimum=min(numeric),
        maximum=max(numeric),
        std=float(pstdev(numeric)),
    )


def measure_prefill_decode(
    model: Any,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> PrefillDecodeRun:
    """Measure one cached greedy generation run.

    Throughput counts generated tokens across the complete batch. TTFT is the first
    model forward pass over the prompt; TPOT is the mean of subsequent decode forward
    passes and is zero when only one token is generated.
    """

    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch_size, sequence_length].")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1.")

    begin_memory_measurement(input_ids.device)
    output = greedy_decode(
        model=model,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
    )
    memory = end_memory_measurement(input_ids.device)

    batch_size = int(input_ids.shape[0])
    generated_per_sequence = int(output.generated_token_ids.shape[1])
    total_generated = batch_size * generated_per_sequence
    decode_total = float(sum(output.decode_times_seconds))
    mean_tpot = (
        decode_total / len(output.decode_times_seconds)
        if output.decode_times_seconds
        else 0.0
    )
    throughput = (
        total_generated / output.total_time_seconds
        if output.total_time_seconds > 0
        else 0.0
    )

    return PrefillDecodeRun(
        batch_size=batch_size,
        prompt_tokens_per_sequence=int(input_ids.shape[1]),
        generated_tokens_per_sequence=generated_per_sequence,
        total_generated_tokens=total_generated,
        target_forward_calls=output.target_forward_calls,
        ttft_seconds=float(output.prefill_time_seconds),
        mean_tpot_seconds=float(mean_tpot),
        decode_total_seconds=decode_total,
        total_latency_seconds=float(output.total_time_seconds),
        throughput_tokens_per_second=float(throughput),
        peak_allocated_bytes=int(memory.get("peak_allocated_bytes", 0) or 0),
        peak_reserved_bytes=int(memory.get("peak_reserved_bytes", 0) or 0),
        raw_output=output,
    )


def benchmark_prefill_decode(
    model: Any,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    warmup_runs: int = 1,
    measured_runs: int = 5,
) -> PrefillDecodeBenchmark:
    """Warm up and repeatedly benchmark one exact token-shape case."""

    if warmup_runs < 0:
        raise ValueError("warmup_runs cannot be negative.")
    if measured_runs < 1:
        raise ValueError("measured_runs must be at least 1.")

    for _ in range(warmup_runs):
        greedy_decode(
            model=model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )

    runs = [
        measure_prefill_decode(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )
        for _ in range(measured_runs)
    ]

    return PrefillDecodeBenchmark(
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
        batch_size=int(input_ids.shape[0]),
        prompt_tokens_per_sequence=int(input_ids.shape[1]),
        requested_new_tokens=max_new_tokens,
        runs=runs,
        ttft_seconds=_distribution([run.ttft_seconds for run in runs]),
        mean_tpot_seconds=_distribution([run.mean_tpot_seconds for run in runs]),
        total_latency_seconds=_distribution(
            [run.total_latency_seconds for run in runs]
        ),
        throughput_tokens_per_second=_distribution(
            [run.throughput_tokens_per_second for run in runs]
        ),
        peak_allocated_bytes=_distribution(
            [float(run.peak_allocated_bytes) for run in runs]
        ),
    )
