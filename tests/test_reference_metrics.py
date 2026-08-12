from __future__ import annotations

import pytest

from src.evaluation.reference_metrics import (
    chrf_score,
    mixed_language_tokens,
    rouge_l_f1,
    score_reference_answer,
    token_f1,
)


def test_identical_reference_metrics_are_one() -> None:
    text = "Blueberry grows in Summer. 蓝莓在夏季生长。"
    report = score_reference_answer(text, text)
    assert report.rouge_l_f1 == pytest.approx(1.0)
    assert report.chrf == pytest.approx(1.0)
    assert report.token_f1 == pytest.approx(1.0)


def test_metrics_detect_partial_fact_overlap() -> None:
    prediction = "Catfish can be caught in rain."
    reference = "Catfish can be caught in a river during rain in Spring or Fall."
    assert 0.0 < rouge_l_f1(prediction, reference) < 1.0
    assert 0.0 < chrf_score(prediction, reference) < 1.0
    assert 0.0 < token_f1(prediction, reference) < 1.0


def test_mixed_language_tokenizer_splits_cjk_characters() -> None:
    tokens = mixed_language_tokens("Corn可以在Summer生长")
    assert "corn" in tokens
    assert "summer" in tokens
    assert "可" in tokens
    assert "以" in tokens
