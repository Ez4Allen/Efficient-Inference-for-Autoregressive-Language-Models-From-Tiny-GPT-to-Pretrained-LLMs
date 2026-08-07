"""Warm, repeated model-pair benchmarking for GameGuideLM.

The benchmark helpers are dependency-light and testable without model weights.
The CLI in ``scripts/benchmark_gameguidelm.py`` owns checkpoint loading and GPU
execution.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from collections import defaultdict
from statistics import mean, median
from typing import Any, Iterable


def sha256_text(value: str) -> str:
    """Return a stable SHA-256 digest for generated text."""

    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def sha256_token_ids(token_ids: Iterable[int]) -> str:
    """Return a stable digest for an ordered generated-token sequence."""

    payload = json.dumps(
        [int(token_id) for token_id in token_ids],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def first_token_mismatch(
    reference: Iterable[int],
    candidate: Iterable[int],
) -> int | None:
    """Return the first differing token position, including length mismatch."""

    reference_ids = [int(token_id) for token_id in reference]
    candidate_ids = [int(token_id) for token_id in candidate]
    common_length = min(len(reference_ids), len(candidate_ids))
    for index in range(common_length):
        if reference_ids[index] != candidate_ids[index]:
            return index
    if len(reference_ids) != len(candidate_ids):
        return common_length
    return None


def token_agreement_rate(
    reference: Iterable[int],
    candidate: Iterable[int],
) -> float:
    """Return position-wise agreement over the longer token sequence."""

    reference_ids = [int(token_id) for token_id in reference]
    candidate_ids = [int(token_id) for token_id in candidate]
    denominator = max(len(reference_ids), len(candidate_ids))
    if denominator == 0:
        return 1.0
    matches = sum(
        left == right
        for left, right in zip(reference_ids, candidate_ids)
    )
    return matches / denominator


def validate_engines(engines: Iterable[str]) -> tuple[str, ...]:
    """Normalize and validate requested benchmark engines.

    Speculative output must be compared with the target output generated from
    the same prompt, so ``target`` is required whenever ``speculative`` is
    requested.
    """

    normalized: list[str] = []
    for engine in engines:
        value = str(engine).strip().casefold()
        if value not in {"target", "draft", "speculative"}:
            raise ValueError(
                "Benchmark engines must be target, draft, or speculative; "
                f"got {engine!r}."
            )
        if value not in normalized:
            normalized.append(value)
    if not normalized:
        raise ValueError("At least one benchmark engine is required.")
    if "speculative" in normalized and "target" not in normalized:
        raise ValueError(
            "The target engine is required when benchmarking speculative "
            "decoding so exact output agreement can be measured."
        )

    # Target must run before speculative so every speculative row has a token
    # reference even when the CLI arguments were supplied in another order.
    canonical_order = ("target", "draft", "speculative")
    return tuple(engine for engine in canonical_order if engine in normalized)


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _metric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0}
    return {
        "mean": round(mean(values), 6),
        "median": round(median(values), 6),
        "p90": round(_percentile(values, 0.90), 6),
    }


def summarize_benchmark_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate warm generation rows by engine.

    Rows marked as warm-up are excluded. Missing optional metrics do not enter
    the corresponding aggregate.
    """

    measured = [dict(row) for row in rows if not bool(row.get("warmup", False))]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in measured:
        grouped[str(row.get("engine") or "unknown")].append(row)

    engines: dict[str, Any] = {}
    for engine, engine_rows in sorted(grouped.items()):
        def values(field: str) -> list[float]:
            return [
                float(row[field])
                for row in engine_rows
                if row.get(field) is not None
            ]

        validation_rows = [
            row for row in engine_rows if row.get("grounding_valid") is not None
        ]
        exact_rows = [
            row for row in engine_rows if row.get("exact_target_match") is not None
        ]
        engines[engine] = {
            "runs": len(engine_rows),
            "examples": len({str(row.get("example_id")) for row in engine_rows}),
            "total_time_seconds": _metric_summary(values("total_time_seconds")),
            "ttft_seconds": _metric_summary(values("ttft_seconds")),
            "mean_tpot_seconds": _metric_summary(values("mean_tpot_seconds")),
            "tokens_per_second": _metric_summary(values("tokens_per_second")),
            "prompt_tokens": _metric_summary(values("prompt_tokens")),
            "generated_tokens": _metric_summary(values("generated_tokens")),
            "target_forward_calls": _metric_summary(values("target_forward_calls")),
            "draft_forward_calls": _metric_summary(values("draft_forward_calls")),
            "acceptance_rate": _metric_summary(values("acceptance_rate")),
            "grounding_valid_rate": round(
                mean(float(bool(row["grounding_valid"])) for row in validation_rows),
                6,
            ) if validation_rows else None,
            "exact_target_match_rate": round(
                mean(float(bool(row["exact_target_match"])) for row in exact_rows),
                6,
            ) if exact_rows else None,
            "token_agreement_rate": _metric_summary(values("token_agreement_rate")),
        }

    target_rows = grouped.get("target", [])
    target_hashes_by_example: dict[str, set[str]] = defaultdict(set)
    for row in target_rows:
        token_hash = row.get("token_ids_sha256")
        if token_hash is not None:
            target_hashes_by_example[str(row.get("example_id"))].add(
                str(token_hash)
            )
    target_deterministic = (
        all(len(hashes) == 1 for hashes in target_hashes_by_example.values())
        if target_hashes_by_example
        else None
    )

    target_mean = (
        engines.get("target", {})
        .get("total_time_seconds", {})
        .get("mean", 0.0)
    )
    speculative_mean = (
        engines.get("speculative", {})
        .get("total_time_seconds", {})
        .get("mean", 0.0)
    )
    speedup = (
        target_mean / speculative_mean
        if target_mean > 0.0 and speculative_mean > 0.0
        else None
    )
    return {
        "measured_rows": len(measured),
        "engines": engines,
        "target_over_speculative_speedup": (
            round(speedup, 6) if speedup is not None else None
        ),
        "target_token_deterministic": target_deterministic,
    }


def basic_environment_metadata() -> dict[str, Any]:
    """Return environment metadata that does not require model loading."""

    payload: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch

        payload.update(
            {
                "torch": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": torch.version.cuda,
            }
        )
        if torch.cuda.is_available():
            payload["gpu"] = torch.cuda.get_device_name(0)
            properties = torch.cuda.get_device_properties(0)
            payload["gpu_memory_bytes"] = int(properties.total_memory)
    except Exception as error:  # pragma: no cover - defensive metadata path
        payload["torch_error"] = f"{type(error).__name__}: {error}"

    try:
        import transformers

        payload["transformers"] = transformers.__version__
    except Exception:
        payload["transformers"] = None
    return payload
