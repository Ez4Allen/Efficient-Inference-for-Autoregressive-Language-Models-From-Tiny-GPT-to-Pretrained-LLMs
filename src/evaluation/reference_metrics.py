"""Reference-answer text metrics used alongside task-specific fact scoring.

The project-specific pass criterion checks required facts, forbidden errors,
citations, and unsupported numeric claims.  This module adds standard text
similarity measurements so the report does not rely on a single custom metric.

The implementation is dependency-light and Unicode-aware:

* ROUGE-L F1 measures longest-common-subsequence overlap.
* chrF measures character n-gram overlap and works for English and Chinese.
* token F1 provides an interpretable lexical-overlap baseline.

BERTScore remains optional because it requires an additional encoder checkpoint;
``scripts/evaluate_reference_metrics.py`` enables it only when explicitly asked.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

_MIXED_TOKEN_PATTERN = re.compile(
    r"[a-zA-Z]+(?:['’-][a-zA-Z]+)*|\d+(?:\.\d+)?|[\u3400-\u4dbf\u4e00-\u9fff]"
)


def normalize_reference_text(value: str) -> str:
    """Return normalized Unicode text while preserving semantic boundaries."""

    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = text.replace("’", "'").replace("‘", "'")
    return " ".join(text.split())


def mixed_language_tokens(value: str) -> list[str]:
    """Tokenize Latin words/numbers and individual CJK characters.

    Whitespace tokenization makes a complete Chinese sentence one token.  This
    mixed tokenizer preserves conventional word tokens for English while using
    character units for CJK text, making the lexical metrics comparable across
    the bilingual GameGuideLM evaluation.
    """

    return _MIXED_TOKEN_PATTERN.findall(normalize_reference_text(value))


def _f_beta(precision: float, recall: float, *, beta: float) -> float:
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    beta_squared = beta * beta
    return (1.0 + beta_squared) * precision * recall / (
        beta_squared * precision + recall
    )


def _lcs_length(first: Sequence[str], second: Sequence[str]) -> int:
    if not first or not second:
        return 0
    # Keep only one dynamic-programming row.  Reference answers are short, but
    # this prevents accidental quadratic memory use on a verbose generation.
    if len(first) < len(second):
        shorter, longer = first, second
    else:
        shorter, longer = second, first
    previous = [0] * (len(shorter) + 1)
    for item in longer:
        current = [0]
        for index, candidate in enumerate(shorter, start=1):
            if item == candidate:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    """Compute ROUGE-L F1 over mixed English/CJK tokens."""

    predicted_tokens = mixed_language_tokens(prediction)
    reference_tokens = mixed_language_tokens(reference)
    if not predicted_tokens or not reference_tokens:
        return 0.0
    lcs = _lcs_length(predicted_tokens, reference_tokens)
    precision = lcs / len(predicted_tokens)
    recall = lcs / len(reference_tokens)
    return _f_beta(precision, recall, beta=1.0)


def token_f1(prediction: str, reference: str) -> float:
    """Compute multiset token F1 using the same bilingual tokenizer."""

    predicted = Counter(mixed_language_tokens(prediction))
    expected = Counter(mixed_language_tokens(reference))
    if not predicted or not expected:
        return 0.0
    overlap = sum((predicted & expected).values())
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(expected.values())
    return _f_beta(precision, recall, beta=1.0)


def _character_ngrams(value: str, order: int) -> Counter[str]:
    text = re.sub(r"\s+", " ", normalize_reference_text(value)).strip()
    if not text or order < 1 or len(text) < order:
        return Counter()
    return Counter(text[index : index + order] for index in range(len(text) - order + 1))


def chrf_score(
    prediction: str,
    reference: str,
    *,
    max_order: int = 6,
    beta: float = 2.0,
) -> float:
    """Compute dependency-free chrF over character n-grams.

    chrF uses recall-heavy F-beta (beta=2 by default) and is a useful standard
    complement for bilingual answers where tokenization varies.  The result is
    returned in [0, 1] rather than the common percentage scale.
    """

    if max_order < 1:
        raise ValueError("max_order must be positive.")
    if beta <= 0:
        raise ValueError("beta must be positive.")

    precisions: list[float] = []
    recalls: list[float] = []
    for order in range(1, max_order + 1):
        predicted = _character_ngrams(prediction, order)
        expected = _character_ngrams(reference, order)
        if not predicted or not expected:
            continue
        overlap = sum((predicted & expected).values())
        precisions.append(overlap / sum(predicted.values()))
        recalls.append(overlap / sum(expected.values()))

    if not precisions:
        return 0.0
    return _f_beta(
        sum(precisions) / len(precisions),
        sum(recalls) / len(recalls),
        beta=beta,
    )


@dataclass(slots=True)
class ReferenceMetricReport:
    rouge_l_f1: float
    chrf: float
    token_f1: float
    prediction_tokens: int
    reference_tokens: int
    bertscore_f1: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_reference_answer(
    prediction: str,
    reference: str,
    *,
    bertscore_f1: float | None = None,
) -> ReferenceMetricReport:
    return ReferenceMetricReport(
        rouge_l_f1=float(rouge_l_f1(prediction, reference)),
        chrf=float(chrf_score(prediction, reference)),
        token_f1=float(token_f1(prediction, reference)),
        prediction_tokens=len(mixed_language_tokens(prediction)),
        reference_tokens=len(mixed_language_tokens(reference)),
        bertscore_f1=(float(bertscore_f1) if bertscore_f1 is not None else None),
    )


def mean_metric(reports: Iterable[ReferenceMetricReport], name: str) -> float:
    values = [getattr(report, name) for report in reports]
    numeric = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    return sum(numeric) / len(numeric) if numeric else 0.0
