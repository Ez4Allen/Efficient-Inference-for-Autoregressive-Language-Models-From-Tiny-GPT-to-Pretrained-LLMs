"""Prompt/output size summaries for reproducible model experiments."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


@dataclass(slots=True)
class DistributionSummary:
    count: int
    minimum: float
    median: float
    p90: float
    p95: float
    p99: float
    maximum: float
    mean: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1].")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_distribution(values: Iterable[float | int]) -> DistributionSummary:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return DistributionSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return DistributionSummary(
        count=len(numeric),
        minimum=min(numeric),
        median=percentile(numeric, 0.50),
        p90=percentile(numeric, 0.90),
        p95=percentile(numeric, 0.95),
        p99=percentile(numeric, 0.99),
        maximum=max(numeric),
        mean=sum(numeric) / len(numeric),
    )


def audit_size_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    group_field: str = "condition",
    prompt_field: str = "prompt_tokens",
    answer_field: str = "generated_tokens",
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        key = str(row.get(group_field) or "all")
        grouped.setdefault(key, []).append(row)

    result: dict[str, Any] = {}
    for key, group in sorted(grouped.items()):
        result[key] = {
            "prompt_tokens": summarize_distribution(
                [row.get(prompt_field) for row in group if row.get(prompt_field) is not None]
            ).to_dict(),
            "answer_tokens": summarize_distribution(
                [row.get(answer_field) for row in group if row.get(answer_field) is not None]
            ).to_dict(),
        }
    return result
