"""Game-agnostic schemas for grounded game-guide answers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class GameEvidence:
    source_id: str
    game: str
    evidence_type: str
    source_catalog_id: str
    label: str
    source_url: str | None = None
    page_title: str | None = None
    section_title: str | None = None
    revision_id: int | None = None
    game_version: str | None = None
    platform: str | None = None
    score: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GameGuideResult:
    game: str
    status: str
    question: str
    intent: str
    entity: str | None
    answer: str
    facts: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[GameEvidence] = field(default_factory=list)
    context_payload: dict[str, Any] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_debug: bool = False) -> dict[str, Any]:
        payload = {
            "game": self.game,
            "status": self.status,
            "question": self.question,
            "intent": self.intent,
            "entity": self.entity,
            "answer": self.answer,
            "facts": self.facts,
            "warnings": self.warnings,
            "candidates": self.candidates,
            "evidence": [item.to_dict() for item in self.evidence],
            "context_payload": self.context_payload,
        }
        if include_debug:
            payload["debug"] = self.debug
        return payload
