"""Evidence selection and prompt compaction for model-facing contexts.

The knowledge plug-ins may return many provenance records and long guide chunks.
This module turns those results into a bounded, source-preserving evidence view
before the prompt reaches Qwen. The selection step is deliberately extractive:
it never invents facts and it never asks another model to summarize evidence.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Literal

from .schemas import GameEvidence, GameGuideResult

EvidencePolicy = Literal["compact", "full", "structured_only", "guide_only"]

_GUIDE_TYPES = {"guide", "guide_chunk", "document", "wiki_chunk"}
_DROP_KEYS = {
    "record",
    "record_json",
    "raw",
    "raw_fields",
    "raw_value",
    "plain_text",
    "context_payload",
    "debug",
}


@dataclass(frozen=True, slots=True)
class EvidenceSelectionConfig:
    """Bound the evidence presented to the language model.

    ``max_characters`` is a reproducible character budget rather than a model-
    specific token budget. The report exposes a conservative token estimate so
    experiments can compare prompt sizes before loading a tokenizer.
    """

    policy: EvidencePolicy = "compact"
    max_sources: int = 6
    max_characters: int = 14_000
    max_characters_per_guide: int = 2_600
    minimum_guide_score: float = 0.0

    def __post_init__(self) -> None:
        if self.policy not in {"compact", "full", "structured_only", "guide_only"}:
            raise ValueError(f"Unsupported evidence policy: {self.policy}")
        if self.max_sources <= 0:
            raise ValueError("max_sources must be positive.")
        if self.max_characters <= 0:
            raise ValueError("max_characters must be positive.")
        if self.max_characters_per_guide <= 0:
            raise ValueError("max_characters_per_guide must be positive.")
        if not 0.0 <= self.minimum_guide_score <= 1.0:
            raise ValueError("minimum_guide_score must be between 0 and 1.")


@dataclass(slots=True)
class EvidenceSelectionReport:
    policy: str
    original_sources: int
    selected_sources: int
    dropped_sources: int
    original_characters: int
    selected_characters: int
    approximate_tokens: int
    budget_exceeded: bool
    selected_source_ids: list[str]
    dropped_source_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PreparedEvidence:
    """A compact copy of a result plus the associated selection report."""

    result: GameGuideResult
    report: EvidenceSelectionReport


def _is_guide(evidence: GameEvidence) -> bool:
    normalized = evidence.evidence_type.casefold().replace("-", "_")
    return normalized in _GUIDE_TYPES or (
        evidence.score is not None and evidence.page_title is not None
    )


def _source_priority(evidence: GameEvidence, ordinal: int) -> tuple[float, int]:
    if _is_guide(evidence):
        score = evidence.score if evidence.score is not None else 0.0
        return (50.0 + float(score), -ordinal)
    type_bonus = {
        "recipe": 12.0,
        "drop": 12.0,
        "npc": 12.0,
        "crop": 12.0,
        "fish": 12.0,
        "villager": 12.0,
        "bundle": 12.0,
        "item": 4.0,
    }.get(evidence.evidence_type.casefold(), 8.0)
    return (100.0 + type_bonus, -ordinal)


def _eligible(evidence: GameEvidence, config: EvidenceSelectionConfig) -> bool:
    guide = _is_guide(evidence)
    if config.policy == "structured_only" and guide:
        return False
    if config.policy == "guide_only" and not guide:
        return False
    if guide and evidence.score is not None:
        if float(evidence.score) < config.minimum_guide_score:
            return False
    return True


def _select_sources(
    evidence: list[GameEvidence],
    config: EvidenceSelectionConfig,
) -> list[GameEvidence]:
    candidates = [item for item in evidence if _eligible(item, config)]
    if config.policy == "full":
        return list(candidates)

    ranked = sorted(
        enumerate(candidates),
        key=lambda pair: _source_priority(pair[1], pair[0]),
        reverse=True,
    )
    selected: list[GameEvidence] = []
    seen_catalog_ids: set[str] = set()
    seen_guide_sections: set[tuple[str | None, str | None]] = set()
    for _, item in ranked:
        if item.source_catalog_id in seen_catalog_ids:
            continue
        if _is_guide(item):
            section_key = (item.page_title, item.section_title)
            if section_key in seen_guide_sections:
                continue
            seen_guide_sections.add(section_key)
        selected.append(item)
        seen_catalog_ids.add(item.source_catalog_id)
        if len(selected) >= config.max_sources:
            break
    original_index = {id(item): index for index, item in enumerate(evidence)}
    return sorted(selected, key=lambda item: original_index[id(item)])


def _truncate_text(value: str, limit: int) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    candidate = text[:limit].rstrip()
    boundary = max(
        candidate.rfind(". "),
        candidate.rfind("? "),
        candidate.rfind("! "),
        candidate.rfind("。"),
        candidate.rfind("；"),
        candidate.rfind("; "),
    )
    if boundary >= max(80, limit // 2):
        candidate = candidate[: boundary + 1].rstrip()
    return candidate + " …"


def _compact_value(value: Any, *, max_string: int = 2_600) -> Any:
    if isinstance(value, str):
        return _truncate_text(value, max_string)
    if isinstance(value, list):
        return [_compact_value(item, max_string=max_string) for item in value]
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).casefold()
            if normalized in _DROP_KEYS:
                continue
            if normalized.startswith("raw_") or normalized.endswith("_json"):
                continue
            compact[key] = _compact_value(item, max_string=max_string)
        return compact
    return value


def _compact_guide_hits(
    facts: dict[str, Any] | None,
    selected: list[GameEvidence],
    *,
    max_characters_per_guide: int,
) -> dict[str, Any] | None:
    if facts is None:
        return None
    compact = _compact_value(deepcopy(facts), max_string=max_characters_per_guide)
    if not isinstance(compact, dict):
        return compact
    hits = compact.get("hits")
    if not isinstance(hits, list):
        return compact

    selected_ids = {item.source_catalog_id for item in selected}
    selected_urls = {item.source_url for item in selected if item.source_url}
    filtered = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        hit_id = str(hit.get("chunk_id") or hit.get("source_catalog_id") or "")
        hit_url = hit.get("source_url")
        if selected_ids and hit_id not in selected_ids and hit_url not in selected_urls:
            continue
        cleaned = _compact_value(hit, max_string=max_characters_per_guide)
        if isinstance(cleaned, dict) and isinstance(cleaned.get("text"), str):
            cleaned["text"] = _truncate_text(cleaned["text"], max_characters_per_guide)
        filtered.append(cleaned)
    compact["hits"] = filtered
    return compact


def _evidence_dict(item: GameEvidence, *, max_string: int) -> dict[str, Any]:
    payload = {
        "source_id": item.source_id,
        "citation": f"[{item.source_id}]",
        "evidence_type": item.evidence_type,
        "label": item.label,
        "source_catalog_id": item.source_catalog_id,
        "source_url": item.source_url,
        "page_title": item.page_title,
        "section_title": item.section_title,
        "game_version": item.game_version,
        "platform": item.platform,
        "score": item.score,
    }
    compact_payload = _compact_value(item.payload, max_string=max_string)
    if compact_payload:
        payload["metadata"] = compact_payload
    return {key: value for key, value in payload.items() if value is not None}


def _serialized_characters(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def prepare_evidence(
    result: GameGuideResult,
    config: EvidenceSelectionConfig | None = None,
) -> PreparedEvidence:
    """Return a compact, non-mutating model-facing result.

    The deterministic answer is retained only on the copied result for fallback
    and validation. Prompt construction decides whether it is exposed to Qwen.
    """

    selected_config = config or EvidenceSelectionConfig()
    original_evidence = list(result.evidence)
    selected = _select_sources(original_evidence, selected_config)

    def build_package(current: list[GameEvidence]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        facts = _compact_guide_hits(
            result.facts,
            current,
            max_characters_per_guide=selected_config.max_characters_per_guide,
        )
        package = {
            "game": result.game,
            "question": result.question,
            "intent": result.intent,
            "entity": result.entity,
            "status": result.status,
            "facts": facts,
            "warnings": list(result.warnings),
            "evidence": [
                _evidence_dict(item, max_string=selected_config.max_characters_per_guide)
                for item in current
            ],
        }
        return facts, package

    compact_facts, package = build_package(selected)
    if _serialized_characters(package) > selected_config.max_characters:
        hits = compact_facts.get("hits") if isinstance(compact_facts, dict) else None
        if isinstance(hits, list) and hits:
            remaining = max(240, selected_config.max_characters // max(1, len(hits)))
            for hit in hits:
                if isinstance(hit, dict) and isinstance(hit.get("text"), str):
                    hit["text"] = _truncate_text(hit["text"], remaining)

    while (
        _serialized_characters(package) > selected_config.max_characters
        and len(selected) > 1
        and any(_is_guide(item) for item in selected)
    ):
        removal_index = max(index for index, item in enumerate(selected) if _is_guide(item))
        selected.pop(removal_index)
        compact_facts, package = build_package(selected)
        hits = compact_facts.get("hits") if isinstance(compact_facts, dict) else None
        if isinstance(hits, list) and hits:
            remaining = max(200, selected_config.max_characters // max(1, len(hits)))
            for hit in hits:
                if isinstance(hit, dict) and isinstance(hit.get("text"), str):
                    hit["text"] = _truncate_text(hit["text"], remaining)

    selected_ids = [item.source_id for item in selected]
    selected_id_set = set(selected_ids)
    dropped_ids = [
        item.source_id for item in original_evidence if item.source_id not in selected_id_set
    ]
    selected_characters = _serialized_characters(package)
    original_package = {
        "facts": result.facts,
        "evidence": [item.to_dict() for item in original_evidence],
    }
    report = EvidenceSelectionReport(
        policy=selected_config.policy,
        original_sources=len(original_evidence),
        selected_sources=len(selected),
        dropped_sources=len(original_evidence) - len(selected),
        original_characters=_serialized_characters(original_package),
        selected_characters=selected_characters,
        approximate_tokens=max(1, math.ceil(selected_characters / 4)),
        budget_exceeded=selected_characters > selected_config.max_characters,
        selected_source_ids=selected_ids,
        dropped_source_ids=dropped_ids,
    )
    prepared = GameGuideResult(
        game=result.game,
        status=result.status,
        question=result.question,
        intent=result.intent,
        entity=result.entity,
        answer=result.answer,
        facts=compact_facts,
        warnings=list(result.warnings),
        candidates=deepcopy(result.candidates),
        evidence=selected,
        context_payload=package,
        debug=deepcopy(result.debug),
    )
    return PreparedEvidence(result=prepared, report=report)
