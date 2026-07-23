from __future__ import annotations

import pytest

from src.optimization import (
    RequestSpec,
    ServingSimulator,
    SimulationConfig,
    generate_poisson_workload,
)


def test_fcfs_single_worker_timeline() -> None:
    simulator = ServingSimulator(
        SimulationConfig(
            num_prefill_workers=1,
            num_decode_workers=1,
            prefill_tokens_per_second=1000.0,
            decode_tokens_per_second=100.0,
            policy_name="fcfs",
        )
    )
    result = simulator.run(
        [
            RequestSpec("a", 0.0, 100, 10),
            RequestSpec("b", 50.0, 50, 5),
        ]
    )

    by_id = {request.request_id: request for request in result.requests}
    assert by_id["a"].prefill_start_ms == pytest.approx(0.0)
    assert by_id["a"].prefill_end_ms == pytest.approx(100.0)
    assert by_id["a"].completion_ms == pytest.approx(200.0)
    assert by_id["b"].prefill_start_ms == pytest.approx(100.0)
    assert by_id["b"].decode_start_ms == pytest.approx(200.0)
    assert by_id["b"].completion_ms == pytest.approx(250.0)


def test_more_decode_workers_reduce_makespan() -> None:
    requests = [
        RequestSpec(f"r{i}", 0.0, 10, 100)
        for i in range(4)
    ]
    one_worker = ServingSimulator(
        SimulationConfig(
            num_decode_workers=1,
            prefill_tokens_per_second=100_000.0,
            decode_tokens_per_second=100.0,
        )
    ).run(requests)
    two_workers = ServingSimulator(
        SimulationConfig(
            num_decode_workers=2,
            prefill_tokens_per_second=100_000.0,
            decode_tokens_per_second=100.0,
        )
    ).run(requests)

    assert two_workers.summary.makespan_ms < one_worker.summary.makespan_ms


def test_shortest_output_policy_changes_decode_order() -> None:
    requests = [
        RequestSpec("long", 0.0, 1, 100),
        RequestSpec("short", 0.0, 1, 1),
    ]
    result = ServingSimulator(
        SimulationConfig(
            prefill_tokens_per_second=1_000_000.0,
            decode_tokens_per_second=100.0,
            policy_name="shortest_output_first",
        )
    ).run(requests)
    by_id = {request.request_id: request for request in result.requests}
    assert by_id["short"].decode_start_ms < by_id["long"].decode_start_ms


def test_poisson_workload_is_reproducible() -> None:
    kwargs = dict(
        num_requests=5,
        arrival_rate_per_second=2.0,
        prompt_lengths=[64, 128],
        output_lengths=[16, 32],
        seed=7,
    )
    assert generate_poisson_workload(**kwargs) == generate_poisson_workload(**kwargs)
