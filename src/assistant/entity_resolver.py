"""Entity validation and clarification for routed Terraria questions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.knowledge.catalog_store import normalize_catalog_name
from src.knowledge.terraria_fact_service import TerrariaFactService

from .entity_aliases import resolve_entity_alias
from .schemas import AssistantIntent, RouteDecision


class EntityResolver:
    """Validate routed entities against the catalog without hallucinating."""

    def __init__(self, service: TerrariaFactService, *, suggestion_limit: int = 5) -> None:
        self.service = service
        self.suggestion_limit = max(1, min(int(suggestion_limit), 20))

    @staticmethod
    def _candidate(
        *,
        entity_type: str,
        name: str,
        source_catalog_id: str | None,
        item_id: int | None = None,
        npc_id: int | None = None,
        internal_name: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "entity_type": entity_type,
            "name": name,
            "source_catalog_id": source_catalog_id,
        }
        if item_id is not None:
            payload["item_id"] = item_id
        if npc_id is not None:
            payload["npc_id"] = npc_id
        if internal_name is not None:
            payload["internal_name"] = internal_name
        return payload

    def _clarification(
        self,
        decision: RouteDecision,
        *,
        candidates: list[dict[str, Any]],
        message: str,
        reason_code: str,
    ) -> RouteDecision:
        resolved = deepcopy(decision)
        resolved.candidates = candidates
        resolved.needs_clarification = True
        resolved.clarification_question = message
        resolved.reason_codes.append(reason_code)
        resolved.confidence = min(resolved.confidence, 0.65)
        return resolved

    def _not_found_suggestions(self, decision: RouteDecision) -> RouteDecision:
        if not decision.entity:
            return decision

        entity = decision.entity
        candidates: list[dict[str, Any]] = []

        if decision.intent == AssistantIntent.RECIPE:
            rows = self.service.store.search_recipes(entity, limit=self.suggestion_limit)
            candidates = [
                self._candidate(
                    entity_type="recipe",
                    name=row["result_name"],
                    source_catalog_id=row["source_catalog_id"],
                    item_id=row.get("result_item_id"),
                )
                for row in rows
            ]
        elif decision.intent in {
            AssistantIntent.ITEM,
            AssistantIntent.DROPS_FOR_ITEM,
            AssistantIntent.RECIPES_USING_ITEM,
        }:
            rows = self.service.store.search_items(entity, limit=self.suggestion_limit)
            candidates = [
                self._candidate(
                    entity_type="item",
                    name=row["name"],
                    source_catalog_id=row["source_catalog_id"],
                    item_id=row.get("item_id"),
                    internal_name=row.get("internal_name"),
                )
                for row in rows
            ]
        elif decision.intent in {AssistantIntent.NPC, AssistantIntent.DROPS_FROM_SOURCE}:
            rows = self.service.store.search_npcs(entity, limit=self.suggestion_limit)
            candidates = [
                self._candidate(
                    entity_type="npc",
                    name=row["name"],
                    source_catalog_id=row["source_catalog_id"],
                    npc_id=row.get("npc_id"),
                )
                for row in rows
            ]

        if not candidates:
            return decision

        resolved = deepcopy(decision)
        resolved.candidates = candidates
        resolved.reason_codes.append("catalog_suggestions_available")
        return resolved

    def resolve(self, decision: RouteDecision) -> RouteDecision:
        if not decision.entity or decision.intent in {
            AssistantIntent.UNKNOWN,
            AssistantIntent.GUIDE,
        }:
            return decision

        entity = decision.entity
        canonical_alias = resolve_entity_alias(entity)
        if canonical_alias != entity:
            decision = deepcopy(decision)
            decision.entity = canonical_alias
            decision.reason_codes.append("entity_alias_resolution")
            decision.confidence = max(decision.confidence, 0.90)
            entity = canonical_alias

        if decision.intent == AssistantIntent.SEARCH:
            item_result = self.service.store.get_item(entity, include_record=False)
            npc_result = self.service.store.get_npc(entity, include_record=False)
            recipe_result = self.service.store.get_recipe(
                entity,
                preferred_only=True,
                include_record=False,
            )
            exact_types = []
            if item_result.get("status") in {"found", "ambiguous"}:
                exact_types.append(AssistantIntent.ITEM)
            if npc_result.get("status") in {"found", "family"}:
                exact_types.append(AssistantIntent.NPC)
            if recipe_result.get("status") in {"found", "ambiguous"}:
                exact_types.append(AssistantIntent.RECIPE)

            # An identity question about an entity that is both an Item and a
            # recipe result should return the Item facts. Crafting questions
            # are already routed explicitly to RECIPE.
            selected_intent: AssistantIntent | None = None
            if AssistantIntent.ITEM in exact_types and AssistantIntent.NPC not in exact_types:
                selected_intent = AssistantIntent.ITEM
            elif exact_types == [AssistantIntent.NPC]:
                selected_intent = AssistantIntent.NPC
            elif exact_types == [AssistantIntent.RECIPE]:
                selected_intent = AssistantIntent.RECIPE

            if selected_intent is not None:
                resolved = deepcopy(decision)
                resolved.intent = selected_intent
                resolved.confidence = max(resolved.confidence, 0.86)
                resolved.reason_codes.append("catalog_type_resolution")
                return self.resolve(resolved)
            return decision
        params = decision.parameters

        if decision.intent == AssistantIntent.ITEM:
            result = self.service.item(
                entity,
                item_id=params.get("item_id"),
                internal_name=params.get("internal_name"),
            )
        elif decision.intent == AssistantIntent.NPC:
            result = self.service.npc(entity, npc_id=params.get("npc_id"))
        elif decision.intent == AssistantIntent.RECIPE:
            result = self.service.recipe(
                entity,
                preferred_only=bool(params.get("preferred_only", True)),
            )
        elif decision.intent == AssistantIntent.RECIPES_USING_ITEM:
            result = self.service.recipes_using_item(
                entity,
                item_id=params.get("item_id"),
                internal_name=params.get("internal_name"),
                preferred_only=bool(params.get("preferred_only", True)),
            )
        elif decision.intent == AssistantIntent.DROPS_FOR_ITEM:
            result = self.service.drops_for_item(
                entity,
                item_id=params.get("item_id"),
                internal_name=params.get("internal_name"),
                mode=str(params.get("mode", "normal")),
                include_partial=True,
            )
        elif decision.intent == AssistantIntent.DROPS_FROM_SOURCE:
            result = self.service.drops_from_source(
                entity,
                mode=str(params.get("mode", "normal")),
                include_partial=True,
            )
        else:
            return decision

        status = result.get("status")
        if status in {"ambiguous", "family"}:
            candidates = list(result.get("candidates") or [])
            if status == "family":
                message = (
                    f"{entity!r} refers to multiple NPC catalog records. "
                    "Please provide the NPC ID or select a candidate."
                )
            else:
                message = (
                    f"{entity!r} matches multiple catalog entities. "
                    "Please provide an item ID, internal name, or select a candidate."
                )
            return self._clarification(
                decision,
                candidates=candidates,
                message=message,
                reason_code=f"{status}_entity",
            )

        if status == "not_found":
            return self._not_found_suggestions(decision)

        facts = result.get("facts") or {}
        canonical_name: str | None = None
        if decision.intent == AssistantIntent.ITEM:
            canonical_name = facts.get("name")
        elif decision.intent == AssistantIntent.NPC:
            canonical_name = facts.get("name")
        elif decision.intent == AssistantIntent.RECIPE:
            canonical_name = facts.get("result_name")
        elif decision.intent == AssistantIntent.RECIPES_USING_ITEM:
            canonical_name = (facts.get("item") or {}).get("name")
        elif decision.intent == AssistantIntent.DROPS_FOR_ITEM:
            drops = facts.get("drops") or []
            canonical_name = drops[0].get("item_name") if drops else entity
        elif decision.intent == AssistantIntent.DROPS_FROM_SOURCE:
            drops = facts.get("drops") or []
            canonical_name = drops[0].get("source_name") if drops else entity

        resolved = deepcopy(decision)
        if canonical_name:
            resolved.entity = str(canonical_name)
            if normalize_catalog_name(canonical_name) != normalize_catalog_name(entity):
                resolved.reason_codes.append("canonical_entity_name_applied")
        return resolved
