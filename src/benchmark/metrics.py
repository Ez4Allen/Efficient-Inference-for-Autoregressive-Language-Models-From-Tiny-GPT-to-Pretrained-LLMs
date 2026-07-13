from __future__ import annotations

from dataclasses import dataclass

from src.decoding.autoregressive import AutoregressiveOutput


@dataclass
class GenerationMetrics:
    """
    Performance metrics for one generation run.
    """

    generated_tokens: int
    ttft_seconds: float
    mean_tpot_seconds: float
    total_latency_seconds: float
    tokens_per_second: float
    target_forward_calls: int


def compute_generation_metrics(
    output: AutoregressiveOutput,
) -> GenerationMetrics:
    """
    Convert raw autoregressive decoding timings into benchmark metrics.

    Args:
        output:
            Raw output returned by greedy_decode().

    Returns:
        GenerationMetrics containing TTFT, TPOT, throughput,
        total latency, and target forward-call count.
    """

    generated_tokens = output.generated_token_ids.shape[1]

    if generated_tokens <= 0:
        raise ValueError("No tokens were generated.")

    ttft_seconds = output.prefill_time_seconds

    if output.decode_times_seconds:
        mean_tpot_seconds = (
            sum(output.decode_times_seconds)
            / len(output.decode_times_seconds)
        )
    else:
        mean_tpot_seconds = 0.0

    total_latency_seconds = output.total_time_seconds

    if total_latency_seconds <= 0:
        raise ValueError("Total latency must be greater than zero.")

    tokens_per_second = (
        generated_tokens / total_latency_seconds
    )

    return GenerationMetrics(
        generated_tokens=generated_tokens,
        ttft_seconds=ttft_seconds,
        mean_tpot_seconds=mean_tpot_seconds,
        total_latency_seconds=total_latency_seconds,
        tokens_per_second=tokens_per_second,
        target_forward_calls=output.target_forward_calls,
    )


def print_generation_metrics(
    metrics: GenerationMetrics,
) -> None:
    """
    Print generation metrics in a readable format.
    """

    print("\n=== Generation Metrics ===")
    print(f"Generated tokens: {metrics.generated_tokens}")
    print(
        f"TTFT: "
        f"{metrics.ttft_seconds * 1000:.3f} ms"
    )
    print(
        f"Mean TPOT: "
        f"{metrics.mean_tpot_seconds * 1000:.3f} ms/token"
    )
    print(
        f"Total latency: "
        f"{metrics.total_latency_seconds:.4f} s"
    )
    print(
        f"Throughput: "
        f"{metrics.tokens_per_second:.2f} tokens/s"
    )
    print(
        f"Target forward calls: "
        f"{metrics.target_forward_calls}"
    )