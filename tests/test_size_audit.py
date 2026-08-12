from __future__ import annotations

import pytest

from src.evaluation.size_audit import audit_size_rows, percentile, summarize_distribution


def test_percentile_and_distribution_summary() -> None:
    values = [1, 2, 3, 4, 5]
    assert percentile(values, 0.5) == 3
    report = summarize_distribution(values)
    assert report.minimum == 1
    assert report.maximum == 5
    assert report.mean == 3


def test_size_audit_groups_conditions() -> None:
    report = audit_size_rows(
        [
            {"condition": "grounded", "prompt_tokens": 100, "generated_tokens": 20},
            {"condition": "grounded", "prompt_tokens": 200, "generated_tokens": 40},
            {"condition": "ungrounded", "prompt_tokens": 20, "generated_tokens": 60},
        ]
    )
    assert report["grounded"]["prompt_tokens"]["maximum"] == 200
    assert report["ungrounded"]["answer_tokens"]["median"] == 60


def test_percentile_rejects_invalid_probability() -> None:
    with pytest.raises(ValueError):
        percentile([1], 1.1)
