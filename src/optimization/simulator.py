"""Deterministic prefill/decode serving simulator."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, Iterable

from .policies import build_policy
from .scheduler import Scheduler


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    arrival_time_ms: float
    prompt_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        if self.arrival_time_ms < 0:
            raise ValueError("arrival_time_ms cannot be negative.")
        if self.prompt_tokens < 1:
            raise ValueError("prompt_tokens must be at least 1.")
        if self.output_tokens < 1:
            raise ValueError("output_tokens must be at least 1.")


@dataclass(frozen=True)
class SimulationConfig:
    num_prefill_workers: int = 1
    num_decode_workers: int = 1
    prefill_tokens_per_second: float = 8_000.0
    decode_tokens_per_second: float = 80.0
    prefill_fixed_overhead_ms: float = 0.0
    kv_transfer_overhead_ms: float = 0.0
    policy_name: str = "fcfs"

    def __post_init__(self) -> None:
        if self.num_prefill_workers < 1 or self.num_decode_workers < 1:
            raise ValueError("Worker counts must be at least 1.")
        if self.prefill_tokens_per_second <= 0:
            raise ValueError("prefill_tokens_per_second must be positive.")
        if self.decode_tokens_per_second <= 0:
            raise ValueError("decode_tokens_per_second must be positive.")
        if self.prefill_fixed_overhead_ms < 0:
            raise ValueError("prefill_fixed_overhead_ms cannot be negative.")
        if self.kv_transfer_overhead_ms < 0:
            raise ValueError("kv_transfer_overhead_ms cannot be negative.")


@dataclass(frozen=True)
class RequestResult:
    request_id: str
    arrival_time_ms: float
    prompt_tokens: int
    output_tokens: int
    prefill_worker_id: int
    decode_worker_id: int
    prefill_start_ms: float
    prefill_end_ms: float
    decode_start_ms: float
    first_token_ms: float
    completion_ms: float
    prefill_queue_ms: float
    decode_queue_ms: float
    ttft_ms: float
    latency_ms: float


@dataclass(frozen=True)
class SimulationSummary:
    requests: int
    makespan_ms: float
    mean_ttft_ms: float
    p95_ttft_ms: float
    mean_latency_ms: float
    p95_latency_ms: float
    request_throughput_per_second: float
    output_token_throughput_per_second: float
    total_output_tokens: int


@dataclass(frozen=True)
class SimulationResult:
    config: SimulationConfig
    requests: list[RequestResult]
    summary: SimulationSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "requests": [asdict(request) for request in self.requests],
            "summary": asdict(self.summary),
        }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


class ServingSimulator:
    """Simulate separated prefill and decode worker pools."""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        policy = build_policy(config.policy_name)
        self.prefill_scheduler = Scheduler(
            num_workers=config.num_prefill_workers,
            policy=policy,
        )
        self.decode_scheduler = Scheduler(
            num_workers=config.num_decode_workers,
            policy=policy,
        )

    def run(self, requests: Iterable[RequestSpec]) -> SimulationResult:
        request_list = list(requests)
        if not request_list:
            raise ValueError("At least one request is required.")
        if len({request.request_id for request in request_list}) != len(request_list):
            raise ValueError("request_id values must be unique.")

        prefill_jobs = self.prefill_scheduler.schedule(
            request_list,
            ready_time=lambda request: request.arrival_time_ms,
            duration_ms=lambda request: (
                self.config.prefill_fixed_overhead_ms
                + request.prompt_tokens
                / self.config.prefill_tokens_per_second
                * 1000.0
            ),
        )
        prefill_by_id = {job.payload.request_id: job for job in prefill_jobs}

        decode_jobs = self.decode_scheduler.schedule(
            request_list,
            ready_time=lambda request: (
                prefill_by_id[request.request_id].finish_time_ms
                + self.config.kv_transfer_overhead_ms
            ),
            duration_ms=lambda request: (
                request.output_tokens
                / self.config.decode_tokens_per_second
                * 1000.0
            ),
        )
        decode_by_id = {job.payload.request_id: job for job in decode_jobs}
        token_time_ms = 1000.0 / self.config.decode_tokens_per_second

        results: list[RequestResult] = []
        for request in sorted(request_list, key=lambda item: item.request_id):
            prefill = prefill_by_id[request.request_id]
            decode = decode_by_id[request.request_id]
            decode_ready = prefill.finish_time_ms + self.config.kv_transfer_overhead_ms
            first_token = decode.start_time_ms + token_time_ms
            results.append(
                RequestResult(
                    request_id=request.request_id,
                    arrival_time_ms=request.arrival_time_ms,
                    prompt_tokens=request.prompt_tokens,
                    output_tokens=request.output_tokens,
                    prefill_worker_id=prefill.worker_id,
                    decode_worker_id=decode.worker_id,
                    prefill_start_ms=prefill.start_time_ms,
                    prefill_end_ms=prefill.finish_time_ms,
                    decode_start_ms=decode.start_time_ms,
                    first_token_ms=first_token,
                    completion_ms=decode.finish_time_ms,
                    prefill_queue_ms=prefill.start_time_ms - request.arrival_time_ms,
                    decode_queue_ms=decode.start_time_ms - decode_ready,
                    ttft_ms=first_token - request.arrival_time_ms,
                    latency_ms=decode.finish_time_ms - request.arrival_time_ms,
                )
            )

        start_time = min(request.arrival_time_ms for request in request_list)
        completion_time = max(result.completion_ms for result in results)
        makespan = completion_time - start_time
        total_output_tokens = sum(request.output_tokens for request in request_list)
        ttfts = [result.ttft_ms for result in results]
        latencies = [result.latency_ms for result in results]

        summary = SimulationSummary(
            requests=len(results),
            makespan_ms=makespan,
            mean_ttft_ms=mean(ttfts),
            p95_ttft_ms=_percentile(ttfts, 0.95),
            mean_latency_ms=mean(latencies),
            p95_latency_ms=_percentile(latencies, 0.95),
            request_throughput_per_second=(
                len(results) / (makespan / 1000.0) if makespan > 0 else 0.0
            ),
            output_token_throughput_per_second=(
                total_output_tokens / (makespan / 1000.0)
                if makespan > 0
                else 0.0
            ),
            total_output_tokens=total_output_tokens,
        )
        return SimulationResult(config=self.config, requests=results, summary=summary)


def generate_poisson_workload(
    *,
    num_requests: int,
    arrival_rate_per_second: float,
    prompt_lengths: list[int],
    output_lengths: list[int],
    seed: int = 42,
) -> list[RequestSpec]:
    """Generate a deterministic Poisson-arrival workload."""

    if num_requests < 1:
        raise ValueError("num_requests must be at least 1.")
    if arrival_rate_per_second <= 0:
        raise ValueError("arrival_rate_per_second must be positive.")
    if not prompt_lengths or any(length < 1 for length in prompt_lengths):
        raise ValueError("prompt_lengths must contain positive integers.")
    if not output_lengths or any(length < 1 for length in output_lengths):
        raise ValueError("output_lengths must contain positive integers.")

    rng = random.Random(seed)
    arrival_ms = 0.0
    requests = []
    for index in range(num_requests):
        if index > 0:
            arrival_ms += rng.expovariate(arrival_rate_per_second) * 1000.0
        requests.append(
            RequestSpec(
                request_id=f"request_{index:05d}",
                arrival_time_ms=arrival_ms,
                prompt_tokens=rng.choice(prompt_lengths),
                output_tokens=rng.choice(output_lengths),
            )
        )
    return requests
