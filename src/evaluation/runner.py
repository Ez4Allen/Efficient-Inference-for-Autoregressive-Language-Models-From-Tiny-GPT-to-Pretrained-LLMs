from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median, pstdev

import torch

from src.evaluation.metrics import (
    GenerationMetrics,
    compute_generation_metrics,
)
from src.inference.autoregressive import greedy_decode
from src.models.loader import ModelBundle


@dataclass
class MetricSummary:
    """
    Statistical summary for one metric across repeated runs.
    """

    mean: float
    p50: float
    p90: float
    std: float


@dataclass
class BenchmarkSummary:
    """
    Aggregated benchmark results across repeated generation runs.
    """

    runs: int
    generated_tokens: int

    ttft_seconds: MetricSummary
    mean_tpot_seconds: MetricSummary
    total_latency_seconds: MetricSummary
    tokens_per_second: MetricSummary

    mean_target_forward_calls: float


def _percentile(
    values: list[float],
    percentile: float,
) -> float:
    """
    Compute a percentile using linear interpolation.

    Args:
        values:
            Numeric observations.

        percentile:
            Percentile in the range [0, 100].

    Returns:
        Interpolated percentile value.
    """

    if not values:
        raise ValueError("Cannot compute percentile of an empty list.")

    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100.")

    sorted_values = sorted(values)

    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * percentile / 100
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)

    weight = position - lower_index

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]

    return lower_value + weight * (upper_value - lower_value)


def _summarize(values: list[float]) -> MetricSummary:
    """
    Compute mean, P50, P90, and population standard deviation.
    """

    if not values:
        raise ValueError("Cannot summarize an empty list.")

    return MetricSummary(
        mean=mean(values),
        p50=median(values),
        p90=_percentile(values, 90),
        std=pstdev(values),
    )


def run_autoregressive_benchmark(
    bundle: ModelBundle,
    prompt: str,
    max_new_tokens: int = 50,
    warmup_runs: int = 5,
    benchmark_runs: int = 20,
) -> BenchmarkSummary:
    """
    Benchmark greedy autoregressive decoding.

    Args:
        bundle:
            Loaded model, tokenizer, device, and dtype.

        prompt:
            Input text used for generation.

        max_new_tokens:
            Maximum number of generated tokens per run.

        warmup_runs:
            Number of unmeasured warm-up runs.

        benchmark_runs:
            Number of measured benchmark runs.

    Returns:
        BenchmarkSummary containing aggregated statistics.
    """

    if warmup_runs < 0:
        raise ValueError("warmup_runs cannot be negative.")

    if benchmark_runs <= 0:
        raise ValueError("benchmark_runs must be greater than zero.")

    encoded = bundle.tokenizer(
        prompt,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(bundle.device)

    print("=== Warm-up ===")

    for warmup_index in range(warmup_runs):
        _ = greedy_decode(
            model=bundle.model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=bundle.tokenizer.eos_token_id,
        )

        print(
            f"Warm-up run "
            f"{warmup_index + 1}/{warmup_runs} completed."
        )

    if bundle.device.type == "cuda":
        torch.cuda.synchronize(bundle.device)

    print("\n=== Benchmark ===")

    run_metrics: list[GenerationMetrics] = []

    for run_index in range(benchmark_runs):
        output = greedy_decode(
            model=bundle.model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=bundle.tokenizer.eos_token_id,
        )

        metrics = compute_generation_metrics(output)
        run_metrics.append(metrics)

        print(
            f"Run {run_index + 1}/{benchmark_runs}: "
            f"TTFT={metrics.ttft_seconds * 1000:.3f} ms, "
            f"TPOT={metrics.mean_tpot_seconds * 1000:.3f} ms, "
            f"Throughput={metrics.tokens_per_second:.2f} tok/s"
        )

    generated_token_counts = {
        metrics.generated_tokens
        for metrics in run_metrics
    }

    if len(generated_token_counts) != 1:
        raise RuntimeError(
            "The number of generated tokens differs across runs. "
            "This may be caused by early EOS termination."
        )

    generated_tokens = run_metrics[0].generated_tokens

    return BenchmarkSummary(
        runs=benchmark_runs,
        generated_tokens=generated_tokens,
        ttft_seconds=_summarize(
            [metrics.ttft_seconds for metrics in run_metrics]
        ),
        mean_tpot_seconds=_summarize(
            [metrics.mean_tpot_seconds for metrics in run_metrics]
        ),
        total_latency_seconds=_summarize(
            [metrics.total_latency_seconds for metrics in run_metrics]
        ),
        tokens_per_second=_summarize(
            [metrics.tokens_per_second for metrics in run_metrics]
        ),
        mean_target_forward_calls=mean(
            [
                metrics.target_forward_calls
                for metrics in run_metrics
            ]
        ),
    )


def print_benchmark_summary(
    summary: BenchmarkSummary,
) -> None:
    """
    Print aggregated benchmark results.
    """

    print("\n=== Benchmark Summary ===")
    print(f"Measured runs: {summary.runs}")
    print(f"Generated tokens per run: {summary.generated_tokens}")

    print("\nTTFT")
    print(
        f"  Mean: {summary.ttft_seconds.mean * 1000:.3f} ms"
    )
    print(
        f"  P50:  {summary.ttft_seconds.p50 * 1000:.3f} ms"
    )
    print(
        f"  P90:  {summary.ttft_seconds.p90 * 1000:.3f} ms"
    )
    print(
        f"  Std:  {summary.ttft_seconds.std * 1000:.3f} ms"
    )

    print("\nMean TPOT")
    print(
        f"  Mean: {summary.mean_tpot_seconds.mean * 1000:.3f} ms/token"
    )
    print(
        f"  P50:  {summary.mean_tpot_seconds.p50 * 1000:.3f} ms/token"
    )
    print(
        f"  P90:  {summary.mean_tpot_seconds.p90 * 1000:.3f} ms/token"
    )
    print(
        f"  Std:  {summary.mean_tpot_seconds.std * 1000:.3f} ms/token"
    )

    print("\nTotal latency")
    print(
        f"  Mean: {summary.total_latency_seconds.mean:.4f} s"
    )
    print(
        f"  P50:  {summary.total_latency_seconds.p50:.4f} s"
    )
    print(
        f"  P90:  {summary.total_latency_seconds.p90:.4f} s"
    )
    print(
        f"  Std:  {summary.total_latency_seconds.std:.4f} s"
    )

    print("\nThroughput")
    print(
        f"  Mean: {summary.tokens_per_second.mean:.2f} tokens/s"
    )
    print(
        f"  P50:  {summary.tokens_per_second.p50:.2f} tokens/s"
    )
    print(
        f"  P90:  {summary.tokens_per_second.p90:.2f} tokens/s"
    )
    print(
        f"  Std:  {summary.tokens_per_second.std:.2f} tokens/s"
    )

    print(
        "\nMean target forward calls: "
        f"{summary.mean_target_forward_calls:.2f}"
    )