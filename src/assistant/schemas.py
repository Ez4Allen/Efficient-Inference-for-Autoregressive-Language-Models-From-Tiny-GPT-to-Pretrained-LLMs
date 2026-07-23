"""Public schemas for the grounded Terraria assistant."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class AssistantIntent(str, Enum):
    """Intents supported by the structured Terraria assistant."""

    ITEM = "item"
    NPC = "npc"
    RECIPE = "recipe"
    RECIPES_USING_ITEM = "recipes_using_item"
    DROPS_FOR_ITEM = "drops_for_item"
    DROPS_FROM_SOURCE = "drops_from_source"
    SEARCH = "search"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class AssistantRequest:
    """A normalized user request passed into :class:`TerrariaAssistant`."""

    question: str
    mode: str = "normal"
    preferred_only: bool = True
    include_partial: bool = True
    language: str = "auto"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.question = str(self.question).strip()
        if not self.question:
            raise ValueError("Assistant question cannot be empty.")

        self.mode = str(self.mode).strip().casefold()
        if self.mode not in {"normal", "expert", "master"}:
            raise ValueError("mode must be one of: normal, expert, master.")

        self.language = str(self.language).strip().casefold()
        if self.language not in {"auto", "en", "zh"}:
            raise ValueError("language must be one of: auto, en, zh.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RouteDecision:
    """Intent and entity selected by the routing layer."""

    intent: AssistantIntent
    entity: str | None
    confidence: float
    parameters: dict[str, Any] = field(default_factory=dict)
    reason_codes: list[str] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: str | None = None

    def __post_init__(self) -> None:
        self.confidence = float(self.confidence)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Route confidence must be between 0 and 1.")

        if self.entity is not None:
            self.entity = str(self.entity).strip() or None

        if self.needs_clarification and not self.clarification_question:
            raise ValueError(
                "A clarification question is required when "
                "needs_clarification is True."
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["intent"] = self.intent.value
        return payload


@dataclass(slots=True)
class ContextBundle:
    """Compact grounded context suitable for an LLM or API client."""

    intent: AssistantIntent
    entity: str | None
    language: str
    text: str
    payload: dict[str, Any]
    evidence: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "entity": self.entity,
            "language": self.language,
            "text": self.text,
            "payload": self.payload,
            "evidence": self.evidence,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class AssistantResponse:
    """Unified public response returned by the Terraria assistant."""

    status: str
    question: str
    answer: str
    intent: AssistantIntent
    entity: str | None
    facts: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    route: RouteDecision | None = None
    context: ContextBundle | None = None
    debug: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.status = str(self.status).strip()
        self.question = str(self.question).strip()
        self.answer = str(self.answer).strip()
        if not self.status:
            raise ValueError("Assistant response status cannot be empty.")
        if not self.question:
            raise ValueError("Assistant response question cannot be empty.")

    def to_dict(
        self,
        *,
        include_debug: bool = False,
        include_context: bool = True,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "question": self.question,
            "answer": self.answer,
            "intent": self.intent.value,
            "entity": self.entity,
            "facts": self.facts,
            "warnings": self.warnings,
            "candidates": self.candidates,
            "evidence": self.evidence,
            "route": self.route.to_dict() if self.route else None,
        }
        if include_context:
            payload["context"] = self.context.to_dict() if self.context else None
        if include_debug:
            payload["debug"] = self.debug
        return payload
