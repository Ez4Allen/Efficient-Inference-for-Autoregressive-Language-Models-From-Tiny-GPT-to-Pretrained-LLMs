from __future__ import annotations

from src.evaluation.model_benchmark import (
    first_token_mismatch,
    sha256_token_ids,
    summarize_benchmark_rows,
    token_agreement_rate,
)


def test_token_sequence_helpers() -> None:
    reference = [1, 2, 3, 4]

    assert first_token_mismatch(reference, reference) is None
    assert first_token_mismatch(reference, [1, 9, 3, 4]) == 1
    assert first_token_mismatch(reference, [1, 2, 3]) == 3
    assert token_agreement_rate(reference, reference) == 1.0
    assert token_agreement_rate(reference, [1, 2, 9, 4]) == 0.75
    assert sha256_token_ids(reference) == sha256_token_ids(tuple(reference))


def test_benchmark_summary_reports_target_determinism_and_agreement() -> None:
    rows = [
        {
            "example_id": "case-1",
            "engine": "target",
            "warmup": False,
            "total_time_seconds": 2.0,
            "ttft_seconds": 0.2,
            "mean_tpot_seconds": 0.1,
            "tokens_per_second": 5.0,
            "prompt_tokens": 10,
            "generated_tokens": 10,
            "target_forward_calls": 10,
            "draft_forward_calls": 0,
            "acceptance_rate": 0.0,
            "grounding_valid": True,
            "exact_target_match": None,
            "token_agreement_rate": None,
            "token_ids_sha256": sha256_token_ids([1, 2, 3]),
        },
        {
            "example_id": "case-1",
            "engine": "target",
            "warmup": False,
            "total_time_seconds": 2.1,
            "ttft_seconds": 0.2,
            "mean_tpot_seconds": 0.1,
            "tokens_per_second": 4.8,
            "prompt_tokens": 10,
            "generated_tokens": 10,
            "target_forward_calls": 10,
            "draft_forward_calls": 0,
            "acceptance_rate": 0.0,
            "grounding_valid": True,
            "exact_target_match": None,
            "token_agreement_rate": None,
            "token_ids_sha256": sha256_token_ids([1, 2, 3]),
        },
        {
            "example_id": "case-1",
            "engine": "speculative",
            "warmup": False,
            "total_time_seconds": 3.0,
            "ttft_seconds": 0.3,
            "mean_tpot_seconds": 0.2,
            "tokens_per_second": 3.3,
            "prompt_tokens": 10,
            "generated_tokens": 10,
            "target_forward_calls": 7,
            "draft_forward_calls": 12,
            "acceptance_rate": 0.5,
            "grounding_valid": True,
            "exact_target_match": False,
            "token_agreement_rate": 0.9,
            "token_ids_sha256": sha256_token_ids([1, 2, 4]),
        },
    ]

    summary = summarize_benchmark_rows(rows)

    assert summary["target_token_deterministic"] is True
    assert summary["engines"]["speculative"]["exact_target_match_rate"] == 0.0
    assert summary["engines"]["speculative"]["token_agreement_rate"]["mean"] == 0.9
