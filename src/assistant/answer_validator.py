"""Conservative validation for grounded LLM answers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .schemas import ContextBundle

_SOURCE_PATTERN = re.compile(r"\[S(\d+)\]")
_URL_PATTERN = re.compile(r"https?://[^\s)\]>]+")


@dataclass(slots=True)
class AnswerValidationResult:
    valid: bool
    citations: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "citations": self.citations,
            "issues": self.issues,
        }


def validate_grounded_answer(
    answer: str,
    context: ContextBundle,
    *,
    require_citations: bool = True,
    max_answer_chars: int = 6000,
) -> AnswerValidationResult:
    issues: list[str] = []
    cleaned = str(answer).strip()
    if not cleaned:
        issues.append("empty_answer")
    if len(cleaned) > int(max_answer_chars):
        issues.append("answer_too_long")
    if "<think" in cleaned.casefold() or "</think" in cleaned.casefold():
        issues.append("thinking_trace_present")

    valid_source_ids = {
        str(item.get("source_id"))
        for item in context.payload.get("evidence", [])
        if item.get("source_id")
    }
    citations = [f"S{value}" for value in _SOURCE_PATTERN.findall(cleaned)]
    invalid_citations = sorted(set(citations) - valid_source_ids)
    if invalid_citations:
        issues.append("invalid_citations:" + ",".join(invalid_citations))
    if require_citations and valid_source_ids and not citations:
        issues.append("missing_citations")

    allowed_urls = {
        str(item.get("source_url"))
        for item in context.payload.get("evidence", [])
        if item.get("source_url")
    }
    emitted_urls = set(_URL_PATTERN.findall(cleaned))
    unsupported_urls = sorted(emitted_urls - allowed_urls)
    if unsupported_urls:
        issues.append("unsupported_urls")

    return AnswerValidationResult(
        valid=not issues,
        citations=citations,
        issues=issues,
    )
