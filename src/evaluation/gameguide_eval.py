"""Evaluation utilities for multi-game grounded language-model experiments."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from src.gameguide.schemas import GameGuideResult
from src.utils.io import read_jsonl, write_json, write_jsonl

_CITATION_PATTERN = re.compile(r"\[(S\d+)\]")


@dataclass(slots=True)
class GameGuideExampleScore:
    example_id: str
    game: str
    language: str
    intent: str
    status_match: bool
    intent_match: bool
    required_fact_coverage: float
    forbidden_error_rate: float
    citation_required: bool
    citation_present: bool
    citation_valid: bool
    citation_precision: float
    evidence_count: int
    selected_evidence_count: int
    fallback_used: bool
    unsupported_numeric_claims: int
    prompt_tokens: int | None
    generated_tokens: int | None
    ttft_seconds: float | None
    total_time_seconds: float | None
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def infer_game_from_path(path: str | Path) -> str | None:
    """Infer the game only when the input path makes it unambiguous."""

    normalized = str(Path(path)).casefold().replace("\\", "/")
    if "terraria" in normalized:
        return "terraria"
    if "stardew" in normalized:
        return "stardew_valley"
    return None


def _infer_language(question: str) -> str:
    return "zh" if any("\u4e00" <= char <= "\u9fff" for char in question) else "en"


def normalize_annotation(
    annotation: dict[str, Any],
    *,
    source_path: str | Path | None = None,
    default_game: str | None = None,
) -> dict[str, Any]:
    """Normalize legacy and game-aware evaluation records without guessing content."""

    record = dict(annotation)
    game = record.get("game") or default_game
    if game is None and source_path is not None:
        game = infer_game_from_path(source_path)
    if game == "stardew":
        game = "stardew_valley"
    if game not in {"terraria", "stardew_valley"}:
        raise ValueError(
            "Evaluation record has no supported game. Add a `game` field, use "
            "a path containing `terraria` or `stardew`, or pass default_game. "
            f"Record id: {record.get('id')!r}; source: {source_path!s}"
        )
    record["game"] = game
    question = str(record.get("question", "")).strip()
    if not question:
        raise ValueError(f"Evaluation record {record.get('id')!r} has no question.")
    record["question"] = question
    record["language"] = str(record.get("language") or _infer_language(question))
    return record


def _fact_text(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("value") or value.get("field") or json.dumps(value, ensure_ascii=False)
    return str(value)


def _normalize_match_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"\b(\d{1,2}):00\s*([ap])\.?m\.?\b", r"\1\2m", text)
    text = re.sub(r"\b(\d{1,2})\s*([ap])\.?m\.?\b", r"\1\2m", text)
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text)


def _semantic_tokens(value: str) -> set[str]:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = text.replace("’", "'").replace("‘", "'")
    return set(re.findall(r"[a-z]+|\d+(?:\.\d+)?|[\u4e00-\u9fff]+", text))


def _contains(text: str, value: Any) -> bool:
    expected = _fact_text(value)
    normalized_expected = _normalize_match_text(expected)
    normalized_answer = _normalize_match_text(text)
    if normalized_expected and normalized_expected in normalized_answer:
        return True

    expected_tokens = _semantic_tokens(expected)
    answer_tokens = _semantic_tokens(text)
    if not expected_tokens:
        return False
    recall = len(expected_tokens & answer_tokens) / len(expected_tokens)
    threshold = 1.0 if len(expected_tokens) <= 4 else 0.75
    return recall >= threshold


def _runtime_from_generation(generation: dict[str, Any]) -> dict[str, Any]:
    runtime = generation.get("runtime")
    if isinstance(runtime, dict):
        return runtime
    attempts = generation.get("attempts")
    if isinstance(attempts, list) and attempts:
        candidate = attempts[-1].get("runtime")
        if isinstance(candidate, dict):
            return candidate
    return {}


def _validation_issues(generation: dict[str, Any]) -> list[str]:
    validation = generation.get("validation")
    if isinstance(validation, dict):
        return [str(item) for item in validation.get("issues") or []]
    attempts = generation.get("attempts")
    if isinstance(attempts, list) and attempts:
        candidate = attempts[-1].get("validation")
        if isinstance(candidate, dict):
            return [str(item) for item in candidate.get("issues") or []]
    return []


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def score_gameguide_result(
    annotation: dict[str, Any],
    result: GameGuideResult,
) -> GameGuideExampleScore:
    required = list(annotation.get("must_include") or annotation.get("required_facts") or [])
    forbidden = list(annotation.get("must_not_include") or annotation.get("forbidden_errors") or [])
    required_hits = sum(_contains(result.answer, item) for item in required)
    forbidden_hits = sum(_contains(result.answer, item) for item in forbidden)
    coverage = required_hits / len(required) if required else 1.0

    expected_status = annotation.get("expected_status") or "found"
    if expected_status in {"not_found", "needs_context", "ambiguous", "partial"}:
        if result.status == expected_status:
            coverage = 1.0
    forbidden_rate = forbidden_hits / len(forbidden) if forbidden else 0.0
    expected_intent = annotation.get("intent")
    status_match = result.status == expected_status
    intent_match = expected_intent is None or result.intent == expected_intent

    generation = result.debug.get("generation") if isinstance(result.debug, dict) else None
    generation = generation if isinstance(generation, dict) else {}
    generated = bool(generation) and not bool(generation.get("skipped", False))
    ungrounded = generation.get("baseline") == "ungrounded"
    fallback_used = bool(generation.get("fallback_used", False))
    citation_required = generated and not ungrounded and result.status == "found" and not fallback_used

    cited_ids = _CITATION_PATTERN.findall(result.answer)
    valid_ids = {item.source_id for item in result.evidence}
    valid_citations = [source_id for source_id in cited_ids if source_id in valid_ids]
    citation_present = bool(cited_ids) if citation_required else True
    citation_valid = (
        bool(cited_ids) and len(valid_citations) == len(cited_ids)
        if citation_required
        else all(source_id in valid_ids for source_id in cited_ids)
    )
    citation_precision = (
        len(valid_citations) / len(cited_ids)
        if cited_ids
        else (0.0 if citation_required else 1.0)
    )

    issues = _validation_issues(generation)
    unsupported_numeric_claims = 0
    for issue in issues:
        if issue.startswith("unsupported_numbers:"):
            values = issue.split(":", 1)[1].strip()
            unsupported_numeric_claims += len([item for item in values.split(",") if item.strip()])

    runtime = _runtime_from_generation(generation)
    evidence_selection = generation.get("evidence_selection")
    selected_evidence_count = len(result.evidence)
    if isinstance(evidence_selection, dict):
        selected_evidence_count = int(
            evidence_selection.get("selected_sources", selected_evidence_count)
        )

    passed = (
        status_match
        and intent_match
        and coverage >= 0.75
        and forbidden_rate == 0.0
        and citation_valid
        and unsupported_numeric_claims == 0
    )
    return GameGuideExampleScore(
        example_id=str(annotation.get("id")),
        game=str(annotation.get("game")),
        language=str(annotation.get("language") or _infer_language(result.question)),
        intent=str(result.intent),
        status_match=status_match,
        intent_match=intent_match,
        required_fact_coverage=round(coverage, 6),
        forbidden_error_rate=round(forbidden_rate, 6),
        citation_required=citation_required,
        citation_present=citation_present,
        citation_valid=citation_valid,
        citation_precision=round(citation_precision, 6),
        evidence_count=len(result.evidence),
        selected_evidence_count=selected_evidence_count,
        fallback_used=fallback_used,
        unsupported_numeric_claims=unsupported_numeric_claims,
        prompt_tokens=_optional_int(runtime.get("prompt_tokens")),
        generated_tokens=_optional_int(runtime.get("generated_tokens")),
        ttft_seconds=_optional_float(runtime.get("ttft_seconds")),
        total_time_seconds=_optional_float(runtime.get("total_time_seconds")),
        passed=passed,
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _summary_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    citation_rows = [row for row in rows if row["score"]["citation_required"]]
    generated_rows = [
        row
        for row in rows
        if (row["result"].get("debug") or {}).get("generation")
    ]
    runtime_rows = [row for row in rows if row["score"]["total_time_seconds"] is not None]
    return {
        "examples": total,
        "pass_rate": _mean([float(row["score"]["passed"]) for row in rows]),
        "status_accuracy": _mean([float(row["score"]["status_match"]) for row in rows]),
        "intent_accuracy": _mean([float(row["score"]["intent_match"]) for row in rows]),
        "mean_required_fact_coverage": _mean(
            [row["score"]["required_fact_coverage"] for row in rows]
        ),
        "mean_forbidden_error_rate": _mean(
            [row["score"]["forbidden_error_rate"] for row in rows]
        ),
        "citation_required_examples": len(citation_rows),
        "citation_presence_rate": _mean(
            [float(row["score"]["citation_present"]) for row in citation_rows]
        ),
        "citation_validity_rate": _mean(
            [float(row["score"]["citation_valid"]) for row in citation_rows]
        ),
        "mean_citation_precision": _mean(
            [row["score"]["citation_precision"] for row in citation_rows]
        ),
        "generated_examples": len(generated_rows),
        "fallback_rate": _mean(
            [float(row["score"]["fallback_used"]) for row in generated_rows]
        ),
        "unsupported_numeric_claims": sum(
            row["score"]["unsupported_numeric_claims"] for row in rows
        ),
        "mean_selected_evidence_count": _mean(
            [float(row["score"]["selected_evidence_count"]) for row in rows]
        ),
        "mean_prompt_tokens": _mean(
            [float(row["score"]["prompt_tokens"]) for row in runtime_rows]
        ),
        "mean_generated_tokens": _mean(
            [float(row["score"]["generated_tokens"]) for row in runtime_rows]
        ),
        "mean_ttft_seconds": _mean(
            [float(row["score"]["ttft_seconds"]) for row in runtime_rows]
        ),
        "mean_generation_seconds": _mean(
            [float(row["score"]["total_time_seconds"]) for row in runtime_rows]
        ),
    }


def _slice_rows(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if field in row["score"]:
            key = str(row["score"].get(field) or "unknown")
        else:
            key = str(row["annotation"].get(field) or "unknown")
        grouped.setdefault(key, []).append(row)
    return {key: _summary_for_rows(group) for key, group in sorted(grouped.items())}


def evaluate_annotations(
    assistant: Any,
    annotations: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for annotation in annotations:
        result = assistant.answer(
            annotation["question"],
            game=annotation["game"],
            player_state=annotation.get("player_state"),
            include_debug=True,
        )
        score = score_gameguide_result(annotation, result)
        rows.append(
            {
                "annotation": annotation,
                "result": result.to_dict(include_debug=True),
                "score": score.to_dict(),
            }
        )
    summary = _summary_for_rows(rows)
    # Slices expose failures hidden by a single average and are central to the
    # multi-game generalization claim.
    summary["slices"] = {
        "game": _slice_rows(rows, "game"),
        "intent": _slice_rows(rows, "intent"),
        "language": _slice_rows(rows, "language"),
    }
    return rows, summary


def evaluate_files(
    assistant: Any,
    input_paths: Iterable[str | Path],
    *,
    output_path: str | Path,
    summary_path: str | Path,
    default_game: str | None = None,
) -> dict[str, Any]:
    annotations: list[dict[str, Any]] = []
    for path in input_paths:
        for record in read_jsonl(Path(path)):
            annotations.append(
                normalize_annotation(
                    record,
                    source_path=path,
                    default_game=default_game,
                )
            )
    rows, summary = evaluate_annotations(assistant, annotations)
    write_jsonl(Path(output_path), rows)
    write_json(Path(summary_path), summary)
    return summary
