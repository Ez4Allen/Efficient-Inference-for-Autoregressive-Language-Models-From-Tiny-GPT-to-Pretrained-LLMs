
from __future__ import annotations

from typing import Any

from src.knowledge.store import StructuredKnowledgeStore


class StructuredFactQueryService:
    """
    Application-level query service for structured Terraria facts.

    Responsibilities:
    - Query the underlying knowledge store.
    - Convert internal database results into a stable response format.
    - Decide whether the result should be answered from structured facts,
      clarified, or passed to another retrieval system.
    """

    def __init__(
        self,
        store: StructuredKnowledgeStore,
    ) -> None:
        self.store = store

    def query(
        self,
        entity_name: str,
        fact_type: str | None = None,
        require_verified: bool = False,
    ) -> dict[str, Any]:
        """
        Query one structured Terraria fact.

        Possible routes:
        - structured_fact: exactly one record was found
        - fallback: no matching structured record was found
        - clarify: multiple records matched
        """
        result = self.store.lookup(
            entity_name=entity_name,
            fact_type=fact_type,
            require_verified=require_verified,
        )

        status = result["status"]

        if status == "not_found":
            return {
                "status": "not_found",
                "route": "fallback",
                "query": entity_name,
                "entity": None,
                "fact_type": fact_type,
                "facts": None,
                "verified": False,
                "source": None,
                "message": (
                    "No exact structured fact was found."
                ),
            }

        if status == "ambiguous":
            candidates = []

            for record in result["candidates"]:
                candidates.append(
                    {
                        "id": record["id"],
                        "entity": record["entity_name"],
                        "fact_type": record["fact_type"],
                        "verified": record.get(
                            "verified",
                            False,
                        ),
                    }
                )

            return {
                "status": "ambiguous",
                "route": "clarify",
                "query": entity_name,
                "entity": None,
                "fact_type": fact_type,
                "facts": None,
                "verified": False,
                "source": None,
                "candidates": candidates,
                "message": (
                    "Multiple structured facts matched "
                    "the supplied entity name."
                ),
            }

        record = result["record"]

        return {
            "status": "found",
            "route": "structured_fact",
            "query": entity_name,
            "entity": record["entity_name"],
            "fact_type": record["fact_type"],
            "facts": self._extract_facts(record),
            "verified": record.get(
                "verified",
                False,
            ),
            "source": record.get("source"),
            "message": (
                "An exact structured fact was found."
            ),
        }

    def _extract_facts(
        self,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Extract fact-type-specific fields from one database record.
        """
        fact_type = record["fact_type"]

        if fact_type == "recipe":
            # Preferred schema: one item may have
            # multiple alternative recipe variants.
            if "recipe_variants" in record:
                return {
                    "schema": "recipe_variants",
                    "recipe_variants": record.get(
                        "recipe_variants",
                        [],
                    ),
                    "notes": record.get(
                        "notes",
                        [],
                    ),
                }

            # Backward-compatible legacy schema.
            return {
                "schema": "legacy_recipe",
                "ingredients": record.get(
                    "ingredients",
                    [],
                ),
                "crafting_stations": record.get(
                    "crafting_stations",
                    [],
                ),
                "conditions": record.get(
                    "conditions",
                    [],
                ),
                "notes": record.get(
                    "notes",
                    [],
                ),
            }

        if fact_type == "boss_summon":
            return {
                "summons": record.get("summons"),
                "usage_conditions": record.get(
                    "usage_conditions",
                    [],
                ),
                "biome_requirements": record.get(
                    "biome_requirements",
                    [],
                ),
                "failure_conditions": record.get(
                    "failure_conditions",
                    [],
                ),
                "notes": record.get(
                    "notes",
                    [],
                ),
            }

        # Generic fallback for fact types added in the future.
        metadata_fields = {
            "id",
            "fact_type",
            "entity_name",
            "aliases",
            "game",
            "scope",
            "platforms",
            "source",
            "verified",
        }

        return {
            key: value
            for key, value in record.items()
            if key not in metadata_fields
        }
