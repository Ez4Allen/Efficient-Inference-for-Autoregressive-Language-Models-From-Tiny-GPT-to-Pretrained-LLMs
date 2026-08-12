"""Diversity and mode-collapse diagnostics for custom language models.

TinyQwenStudent is optimized for teacher alignment, not open-ended creativity.
The metrics here are therefore reported as auxiliary diagnostics:

* Distinct-n and unique-output rate detect repeated/collapsed generations.
* Self-BLEU estimates similarity among multiple samples for one prompt.
* Repetition rate detects local token loops.
* Conditional top-1 diversity measures whether a model predicts the same token
  across heterogeneous contexts.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

from .reference_metrics import mixed_language_tokens


def _ngrams(tokens: Sequence[str], order: int) -> list[tuple[str, ...]]:
    if order < 1:
        raise ValueError("order must be positive.")
    if len(tokens) < order:
        return []
    return [tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1)]


def distinct_n(texts: Iterable[str], *, order: int) -> float:
    all_ngrams: list[tuple[str, ...]] = []
    for text in texts:
        all_ngrams.extend(_ngrams(mixed_language_tokens(text), order))
    return len(set(all_ngrams)) / len(all_ngrams) if all_ngrams else 0.0


def unique_output_rate(texts: Iterable[str]) -> float:
    normalized = [" ".join(mixed_language_tokens(text)) for text in texts]
    normalized = [text for text in normalized if text]
    return len(set(normalized)) / len(normalized) if normalized else 0.0


def repetition_rate(texts: Iterable[str], *, order: int = 3) -> float:
    """Return the fraction of n-gram occurrences that repeat within outputs."""

    total = 0
    repeated = 0
    for text in texts:
        grams = _ngrams(mixed_language_tokens(text), order)
        counts = Counter(grams)
        total += len(grams)
        repeated += sum(max(0, count - 1) for count in counts.values())
    return repeated / total if total else 0.0


def _sentence_bleu(candidate: str, references: Sequence[str], *, max_order: int = 4) -> float:
    candidate_tokens = mixed_language_tokens(candidate)
    reference_tokens = [mixed_language_tokens(reference) for reference in references]
    if not candidate_tokens or not reference_tokens:
        return 0.0

    log_precisions: list[float] = []
    for order in range(1, max_order + 1):
        candidate_counts = Counter(_ngrams(candidate_tokens, order))
        if not candidate_counts:
            # Add-one smoothing for short generated answers.
            log_precisions.append(math.log(1.0 / 2.0))
            continue
        max_reference_counts: Counter[tuple[str, ...]] = Counter()
        for tokens in reference_tokens:
            counts = Counter(_ngrams(tokens, order))
            for gram, count in counts.items():
                max_reference_counts[gram] = max(max_reference_counts[gram], count)
        clipped = sum(
            min(count, max_reference_counts[gram])
            for gram, count in candidate_counts.items()
        )
        precision = (clipped + 1.0) / (sum(candidate_counts.values()) + 1.0)
        log_precisions.append(math.log(precision))

    candidate_length = len(candidate_tokens)
    closest_reference_length = min(
        (len(tokens) for tokens in reference_tokens),
        key=lambda length: (abs(length - candidate_length), length),
    )
    brevity_penalty = (
        1.0
        if candidate_length > closest_reference_length
        else math.exp(1.0 - closest_reference_length / max(1, candidate_length))
    )
    return brevity_penalty * math.exp(sum(log_precisions) / len(log_precisions))


def self_bleu(texts: Sequence[str], *, max_order: int = 4) -> float:
    if len(texts) < 2:
        return 0.0
    scores = []
    for index, candidate in enumerate(texts):
        references = [text for other_index, text in enumerate(texts) if other_index != index]
        scores.append(_sentence_bleu(candidate, references, max_order=max_order))
    return sum(scores) / len(scores)


def conditional_top1_diversity(token_ids: Sequence[int]) -> float:
    """Fraction of unique top-1 predictions across evaluated positions."""

    values = [int(value) for value in token_ids]
    return len(set(values)) / len(values) if values else 0.0


@dataclass(slots=True)
class DiversityReport:
    samples: int
    unique_output_rate: float
    distinct_1: float
    distinct_2: float
    self_bleu_4: float
    trigram_repetition_rate: float
    mean_output_tokens: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_output_diversity(texts: Sequence[str]) -> DiversityReport:
    token_lengths = [len(mixed_language_tokens(text)) for text in texts]
    return DiversityReport(
        samples=len(texts),
        unique_output_rate=float(unique_output_rate(texts)),
        distinct_1=float(distinct_n(texts, order=1)),
        distinct_2=float(distinct_n(texts, order=2)),
        self_bleu_4=float(self_bleu(texts, max_order=4)),
        trigram_repetition_rate=float(repetition_rate(texts, order=3)),
        mean_output_tokens=(sum(token_lengths) / len(token_lengths) if token_lengths else 0.0),
    )
