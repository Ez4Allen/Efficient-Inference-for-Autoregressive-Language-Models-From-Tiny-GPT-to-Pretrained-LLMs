"""Adapter that exposes the existing Terraria assistant as a game plug-in."""

from __future__ import annotations

from typing import Any

from src.assistant import TerrariaAssistant
from src.gameguide.schemas import GameEvidence, GameGuideResult


class TerrariaGamePlugin:
    game_id = "terraria"
    display_name = "Terraria"

    def __init__(self, *, auto_build: bool = True) -> None:
        self.assistant = TerrariaAssistant(auto_build=auto_build, generator=None)

    def close(self) -> None:
        self.assistant.close()

    def answer(
        self,
        question: str,
        *,
        language: str = "auto",
        player_state: dict[str, Any] | None = None,
        include_debug: bool = False,
    ) -> GameGuideResult:
        mode = str((player_state or {}).get("mode", "normal"))
        response = self.assistant.answer(
            question,
            mode=mode,
            language=language,
            include_debug=include_debug,
        )
        evidence = []
        for item in response.evidence:
            evidence.append(
                GameEvidence(
                    source_id=str(item.get("source_id")),
                    game="terraria",
                    evidence_type=str(item.get("entity_type") or "fact"),
                    source_catalog_id=str(item.get("source_catalog_id") or item.get("document_id") or item.get("source_id")),
                    label=str(item.get("page_title") or item.get("entity_name") or item.get("source_catalog_id")),
                    source_url=item.get("source_url"),
                    page_title=item.get("page_title"),
                    section_title=item.get("section_title"),
                    revision_id=item.get("revision_id"),
                    game_version=item.get("game_version"),
                    platform=item.get("platform"),
                    score=item.get("score"),
                    payload=dict(item),
                )
            )
        context_payload = response.context.payload if response.context is not None else {}
        return GameGuideResult(
            game="terraria",
            status=response.status,
            question=response.question,
            intent=response.intent.value,
            entity=response.entity,
            answer=response.answer,
            facts=response.facts,
            warnings=response.warnings,
            candidates=response.candidates,
            evidence=evidence,
            context_payload=context_payload,
            debug=response.debug if include_debug else {},
        )
