"""Structured retrieval adapter over :class:`TerrariaFactService`."""

from __future__ import annotations

from typing import Any

from src.knowledge.terraria_fact_service import TerrariaFactService

from .schemas import AssistantIntent, AssistantRequest, RouteDecision


class StructuredRetriever:
    """Translate a route decision into a deterministic FactService query."""

    def __init__(self, service: TerrariaFactService) -> None:
        self.service = service

    def retrieve(
        self,
        request: AssistantRequest,
        route: RouteDecision,
    ) -> dict[str, Any]:
        entity = route.entity or request.question
        params = dict(route.parameters)

        mode = str(params.get("mode", request.mode))
        preferred_only = bool(params.get("preferred_only", request.preferred_only))
        include_partial = bool(params.get("include_partial", request.include_partial))
        item_id = params.get("item_id")
        npc_id = params.get("npc_id")
        internal_name = params.get("internal_name")

        if route.intent == AssistantIntent.ITEM:
            return self.service.item(
                entity,
                item_id=item_id,
                internal_name=internal_name,
            )
        if route.intent == AssistantIntent.NPC:
            return self.service.npc(entity, npc_id=npc_id)
        if route.intent == AssistantIntent.RECIPE:
            return self.service.recipe(entity, preferred_only=preferred_only)
        if route.intent == AssistantIntent.RECIPES_USING_ITEM:
            return self.service.recipes_using_item(
                entity,
                item_id=item_id,
                internal_name=internal_name,
                preferred_only=preferred_only,
            )
        if route.intent == AssistantIntent.DROPS_FOR_ITEM:
            return self.service.drops_for_item(
                entity,
                item_id=item_id,
                internal_name=internal_name,
                mode=mode,
                include_partial=include_partial,
            )
        if route.intent == AssistantIntent.DROPS_FROM_SOURCE:
            return self.service.drops_from_source(
                entity,
                mode=mode,
                include_partial=include_partial,
            )
        return self.service.search(entity)
