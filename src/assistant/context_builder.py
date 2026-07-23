"""Grounded context construction for deterministic or LLM answer generation."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from .schemas import AssistantIntent, AssistantRequest, ContextBundle, RouteDecision


class ContextBuilder:
    """Create compact evidence payloads without discarding provenance."""

    def __init__(self, *, max_entries: int = 20) -> None:
        self.max_entries = max(1, int(max_entries))

    def _truncate(self, value: Any) -> Any:
        if isinstance(value, list):
            return [self._truncate(item) for item in value[: self.max_entries]]
        if isinstance(value, dict):
            return {key: self._truncate(item) for key, item in value.items()}
        return value

    def build(
        self,
        request: AssistantRequest,
        route: RouteDecision,
        retrieval: dict[str, Any],
        *,
        language: str,
    ) -> ContextBundle:
        facts = self._truncate(deepcopy(retrieval.get("facts")))
        payload = {
            "question": request.question,
            "intent": route.intent.value,
            "entity": route.entity,
            "status": retrieval.get("status"),
            "facts": facts,
            "warnings": list(retrieval.get("warnings") or []),
            "candidates": self._truncate(list(retrieval.get("candidates") or [])),
            "evidence": list(retrieval.get("provenance") or []),
        }
        instructions = (
            "仅根据以下 Terraria 证据回答。不要编造物品、NPC、配方、掉落率或游戏机制。"
            "如果证据不足或存在歧义，请明确说明。"
            if language == "zh"
            else "Answer only from the Terraria evidence below. Do not invent items, "
            "NPCs, recipes, drop rates, or mechanics. State clearly when evidence "
            "is missing or ambiguous."
        )
        text = (
            f"{instructions}\n\n"
            f"Question: {request.question}\n"
            f"Intent: {route.intent.value}\n"
            f"Entity: {route.entity}\n"
            "Evidence JSON:\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )
        return ContextBundle(
            intent=route.intent,
            entity=route.entity,
            language=language,
            text=text,
            payload=payload,
            evidence=list(retrieval.get("provenance") or []),
            warnings=list(retrieval.get("warnings") or []),
        )
