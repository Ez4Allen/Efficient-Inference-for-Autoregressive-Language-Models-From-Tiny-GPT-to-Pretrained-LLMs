"""Conservative validation for multi-game grounded answers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .schemas import GameGuideResult

_SOURCE_PATTERN = re.compile(r"\[S(\d+)\]")
_URL_PATTERN = re.compile(r"https?://[^\s)\]>]+")
_NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?:%|g)?(?![A-Za-z0-9_])"
)
_LIST_NUMBER_PATTERN = re.compile(r"(?m)^\s*\d+[.)]\s+")


@dataclass(slots=True)
class GroundedValidation:
    valid: bool
    citations: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {"valid": self.valid, "citations": self.citations, "issues": self.issues}


def _normalize_number(value: str) -> str:
    normalized = value.casefold().replace(",", "").strip()
    suffix = ""
    if normalized.endswith("%") or normalized.endswith("g"):
        suffix = normalized[-1]
        normalized = normalized[:-1]
    try:
        numeric = float(normalized)
    except ValueError:
        return value.casefold().replace(",", "").strip()
    if numeric.is_integer():
        body = str(int(numeric))
    else:
        body = format(numeric, ".12g")
    return body + suffix


def _extract_numbers(text: str) -> set[str]:
    # Source identifiers and ordered-list markers are formatting, not claims.
    without_citations = _SOURCE_PATTERN.sub("", str(text))
    without_list_numbers = _LIST_NUMBER_PATTERN.sub("", without_citations)
    return {_normalize_number(match.group(0)) for match in _NUMBER_PATTERN.finditer(without_list_numbers)}


def _support_corpus(result: GameGuideResult) -> str:
    payload: dict[str, Any] = {
        "deterministic_answer": result.answer,
        "facts": result.facts,
        "context_payload": result.context_payload,
        "evidence": [item.to_dict() for item in result.evidence],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def validate_gameguide_answer(
    answer: str,
    result: GameGuideResult,
    *,
    require_citations: bool = True,
    validate_numbers: bool = True,
    max_answer_chars: int = 7000,
) -> GroundedValidation:
    """Validate citations and high-risk numeric claims.

    This is deliberately conservative. It does not claim to prove semantic
    correctness; instead it rejects several common failure modes and lets the
    caller return the deterministic evidence answer as a safe fallback.
    """

    text = str(answer).strip()
    issues: list[str] = []
    if not text:
        issues.append("empty_answer")
    if len(text) > int(max_answer_chars):
        issues.append("answer_too_long")
    if "<think" in text.casefold() or "</think" in text.casefold():
        issues.append("thinking_trace_present")

    valid_ids = {item.source_id for item in result.evidence}
    citations = [f"S{value}" for value in _SOURCE_PATTERN.findall(text)]
    invalid = sorted(set(citations) - valid_ids)
    if invalid:
        issues.append("invalid_citations:" + ",".join(invalid))
    if require_citations and valid_ids and not citations:
        issues.append("missing_citations")

    allowed_urls = {item.source_url for item in result.evidence if item.source_url}
    emitted_urls = set(_URL_PATTERN.findall(text))
    if emitted_urls - allowed_urls:
        issues.append("unsupported_urls")

    if validate_numbers and result.status == "found":
        answer_numbers = _extract_numbers(text)
        supported_numbers = _extract_numbers(_support_corpus(result))
        unsupported_numbers = sorted(answer_numbers - supported_numbers)
        if unsupported_numbers:
            issues.append("unsupported_numbers:" + ",".join(unsupported_numbers))

    return GroundedValidation(valid=not issues, citations=citations, issues=issues)
