"""Grounded context construction for deterministic or LLM answer generation."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from .schemas import AssistantIntent, AssistantRequest, ContextBundle, RouteDecision


class ContextBuilder:
    """Create compact evidence payloads without discarding provenance."""

    def __init__(
        self,
        *,
        max_entries: int = 20,
        max_string_chars: int = 4000,
    ) -> None:
        self.max_entries = max(1, int(max_entries))
        self.max_string_chars = max(200, int(max_string_chars))

    def _truncate(self, value: Any) -> Any:
        if isinstance(value, str):
            if len(value) <= self.max_string_chars:
                return value
            return value[: self.max_string_chars - 3].rstrip() + "..."
        if isinstance(value, list):
            return [self._truncate(item) for item in value[: self.max_entries]]
        if isinstance(value, dict):
            return {key: self._truncate(item) for key, item in value.items()}
        return value

    @staticmethod
    def _source_annotated_provenance(
        provenance: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            {"source_id": f"S{index}", **deepcopy(item)}
            for index, item in enumerate(provenance, start=1)
        ]

    @staticmethod
    def _annotate_guide_hits(
        facts: Any,
        evidence: list[dict[str, Any]],
    ) -> Any:
        if not isinstance(facts, dict) or not isinstance(facts.get("hits"), list):
            return facts
        by_chunk_id = {
            item.get("source_catalog_id"): item.get("source_id")
            for item in evidence
            if item.get("source_catalog_id") and item.get("source_id")
        }
        for hit in facts["hits"]:
            if isinstance(hit, dict):
                source_id = by_chunk_id.get(hit.get("chunk_id"))
                if source_id:
                    hit["source_id"] = source_id
        return facts

    def build(
        self,
        request: AssistantRequest,
        route: RouteDecision,
        retrieval: dict[str, Any],
        *,
        language: str,
    ) -> ContextBundle:
        raw_provenance = list(retrieval.get("provenance") or [])
        evidence = self._source_annotated_provenance(raw_provenance)
        facts = self._truncate(deepcopy(retrieval.get("facts")))
        facts = self._annotate_guide_hits(facts, evidence)
        payload = {
            "question": request.question,
            "intent": route.intent.value,
            "entity": route.entity,
            "status": retrieval.get("status"),
            "facts": facts,
            "warnings": list(retrieval.get("warnings") or []),
            "candidates": self._truncate(list(retrieval.get("candidates") or [])),
            "evidence": self._truncate(evidence),
        }
        if route.intent == AssistantIntent.GUIDE:
            instructions = (
                "仅根据以下本地 Terraria 攻略段落回答。不要补充检索证据中没有的进度、"
                "机制、配装或战斗建议；如果证据不足，请明确说明。"
                if language == "zh"
                else "Answer only from the retrieved local Terraria guide excerpts below. "
                "Do not add progression or mechanics advice that is absent from the evidence. "
                "State clearly when the evidence is insufficient."
            )
        else:
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
            evidence=evidence,
            warnings=list(retrieval.get("warnings") or []),
        )
