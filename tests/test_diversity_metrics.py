from __future__ import annotations

from src.evaluation.diversity_metrics import (
    analyze_output_diversity,
    conditional_top1_diversity,
    distinct_n,
    self_bleu,
    unique_output_rate,
)


def test_collapsed_outputs_have_low_unique_rate_and_high_self_bleu() -> None:
    texts = ["same answer here"] * 4
    report = analyze_output_diversity(texts)
    assert report.unique_output_rate == 0.25
    assert report.self_bleu_4 > 0.8


def test_varied_outputs_improve_distinct_metrics() -> None:
    repeated = ["same same same", "same same same"]
    varied = ["blueberry grows summer", "catfish needs rain"]
    assert distinct_n(varied, order=1) > distinct_n(repeated, order=1)
    assert unique_output_rate(varied) == 1.0
    assert self_bleu(varied) < 1.0


def test_conditional_top1_diversity() -> None:
    assert conditional_top1_diversity([1, 1, 1, 1]) == 0.25
    assert conditional_top1_diversity([1, 2, 3, 4]) == 1.0
