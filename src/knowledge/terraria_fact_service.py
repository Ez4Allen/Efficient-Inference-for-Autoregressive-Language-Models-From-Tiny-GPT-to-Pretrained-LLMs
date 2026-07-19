
from __future__ import annotations

import json

from pathlib import Path
from typing import Any

from .terraria_query_store import (
    DEFAULT_DATABASE_PATH,
    TerrariaQueryStore,
)


class TerrariaFactService:
    """
    Higher-level deterministic fact service built on
    top of TerrariaQueryStore.

    The service converts database-oriented results into
    compact payloads suitable for:

    - application APIs
    - retrieval context
    - prompt construction
    - deterministic answer rendering
    """

    VALID_INTENTS = {
        "item",
        "npc",
        "recipe",
        "recipes_using_item",
        "drops_for_item",
        "drops_from_source",
        "search",
    }

    def __init__(
        self,
        database_path: str | Path = (
            DEFAULT_DATABASE_PATH
        ),
    ) -> None:
        self.database_path = Path(
            database_path
        )

        self.store = TerrariaQueryStore(
            self.database_path,
            read_only=True,
        )

        self._closed = False

    def __enter__(
        self,
    ) -> "TerrariaFactService":
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()

    def _ensure_open(
        self,
    ) -> None:
        if self._closed:
            raise RuntimeError(
                "TerrariaFactService is closed."
            )

    def close(
        self,
    ) -> None:
        if not self._closed:
            self.store.close()
            self._closed = True

    @staticmethod
    def _percentage(
        value: float | int | None,
    ) -> float | None:
        if value is None:
            return None

        return round(
            float(value) * 100.0,
            4,
        )

    @classmethod
    def _format_probability(
        cls,
        probability: Any,
    ) -> dict[str, Any] | None:
        if not isinstance(
            probability,
            dict,
        ):
            return None

        minimum = probability.get(
            "minimum"
        )

        maximum = probability.get(
            "maximum"
        )

        if (
            minimum is None
            and maximum is None
        ):
            return None

        if maximum is None:
            maximum = minimum

        if minimum is None:
            minimum = maximum

        minimum_percent = cls._percentage(
            minimum
        )

        maximum_percent = cls._percentage(
            maximum
        )

        if minimum_percent == maximum_percent:
            display = (
                f"{minimum_percent:g}%"
            )

        else:
            display = (
                f"{minimum_percent:g}"
                f"–{maximum_percent:g}%"
            )

        return {
            "minimum": minimum,
            "maximum": maximum,
            "minimum_percent": (
                minimum_percent
            ),
            "maximum_percent": (
                maximum_percent
            ),
            "display": display,
        }

    @staticmethod
    def _format_quantity(
        quantity: Any,
    ) -> dict[str, Any] | None:
        if not isinstance(
            quantity,
            dict,
        ):
            return None

        minimum = quantity.get(
            "minimum"
        )

        maximum = quantity.get(
            "maximum"
        )

        if (
            minimum is None
            and maximum is None
        ):
            return None

        if maximum is None:
            maximum = minimum

        if minimum is None:
            minimum = maximum

        if minimum == maximum:
            display = str(minimum)

        else:
            display = (
                f"{minimum}–{maximum}"
            )

        return {
            "minimum": minimum,
            "maximum": maximum,
            "display": display,
        }

    @staticmethod
    def _unique_provenance(
        entries: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        seen: set[
            tuple[Any, ...]
        ] = set()

        unique_entries = []

        for entry in entries:
            key = (
                entry.get("entity_type"),
                entry.get(
                    "source_catalog_id"
                ),
            )

            if key in seen:
                continue

            seen.add(key)
            unique_entries.append(entry)

        return unique_entries

    @staticmethod
    def _candidate_summary(
        match: dict[str, Any],
        *,
        entity_type: str,
    ) -> dict[str, Any]:
        summary = {
            "entity_type": entity_type,
            "name": match.get("name"),
            "source_catalog_id": (
                match.get(
                    "source_catalog_id"
                )
            ),
        }

        if entity_type == "item":
            summary.update(
                {
                    "item_id": match.get(
                        "item_id"
                    ),
                    "internal_name": (
                        match.get(
                            "internal_name"
                        )
                    ),
                }
            )

        elif entity_type == "npc":
            summary["npc_id"] = match.get(
                "npc_id"
            )

        return summary

    # =================================================
    # Item
    # =================================================

    def item(
        self,
        name: str,
    ) -> dict[str, Any]:
        self._ensure_open()

        result = self.store.get_item(
            name,
            include_record=True,
        )

        if result["status"] == "not_found":
            return {
                "status": "not_found",
                "intent": "item",
                "query": name,
                "facts": None,
                "candidates": [],
                "warnings": [
                    "No exact Item catalog match."
                ],
                "provenance": [],
            }

        if result["status"] == "ambiguous":
            candidates = [
                self._candidate_summary(
                    match,
                    entity_type="item",
                )
                for match
                in result["matches"]
            ]

            return {
                "status": "ambiguous",
                "intent": "item",
                "query": name,
                "facts": None,
                "candidates": candidates,
                "warnings": [
                    "Multiple Items share this name."
                ],
                "provenance": [
                    {
                        "entity_type": "item",
                        "source_catalog_id": (
                            candidate[
                                "source_catalog_id"
                            ]
                        ),
                    }
                    for candidate in candidates
                ],
            }

        match = result["match"]
        record = match.get(
            "record",
            {},
        )

        facts = {
            "name": match["name"],
            "item_id": match["item_id"],
            "internal_name": (
                match["internal_name"]
            ),
            "classification": record.get(
                "classification",
                {},
            ),
            "rarity": record.get(
                "rarity"
            ),
            "stack": record.get(
                "stack"
            ),
            "value": record.get(
                "value",
                {},
            ),
            "parse_status": (
                match["parse_status"]
            ),
        }

        return {
            "status": "found",
            "intent": "item",
            "query": name,
            "facts": facts,
            "candidates": [],
            "warnings": [],
            "provenance": [
                {
                    "entity_type": "item",
                    "source_catalog_id": (
                        match[
                            "source_catalog_id"
                        ]
                    ),
                }
            ],
        }

    # =================================================
    # NPC
    # =================================================

    def npc(
        self,
        name: str,
        *,
        npc_id: int | None = None,
    ) -> dict[str, Any]:
        self._ensure_open()

        result = self.store.get_npc(
            name,
            npc_id=npc_id,
            include_record=True,
        )

        if result["status"] == "not_found":
            return {
                "status": "not_found",
                "intent": "npc",
                "query": name,
                "facts": None,
                "candidates": [],
                "warnings": [
                    "No matching NPC catalog record."
                ],
                "provenance": [],
            }

        if result["status"] == "family":
            candidates = [
                self._candidate_summary(
                    match,
                    entity_type="npc",
                )
                for match
                in result["matches"]
            ]

            return {
                "status": "family",
                "intent": "npc",
                "query": name,
                "facts": {
                    "name": name,
                    "member_count": len(
                        candidates
                    ),
                    "members": candidates,
                },
                "candidates": candidates,
                "warnings": [
                    "The NPC name refers to multiple "
                    "catalog entities or body parts."
                ],
                "provenance": [
                    {
                        "entity_type": "npc",
                        "source_catalog_id": (
                            candidate[
                                "source_catalog_id"
                            ]
                        ),
                    }
                    for candidate in candidates
                ],
            }

        match = result["match"]
        record = match.get(
            "record",
            {},
        )

        facts = {
            "name": match["name"],
            "npc_id": match["npc_id"],
            "npc_types": record.get(
                "npc_types",
                [],
            ),
            "environment": record.get(
                "environment",
                [],
            ),
            "ai": record.get("ai"),
            "stats": record.get(
                "stats",
                {},
            ),
            "immunities": record.get(
                "immunities",
                [],
            ),
            "coin_drop": record.get(
                "coin_drop",
                {},
            ),
            "parse_status": (
                match["parse_status"]
            ),
        }

        return {
            "status": "found",
            "intent": "npc",
            "query": name,
            "facts": facts,
            "candidates": [],
            "warnings": [],
            "provenance": [
                {
                    "entity_type": "npc",
                    "source_catalog_id": (
                        match[
                            "source_catalog_id"
                        ]
                    ),
                }
            ],
        }

    # =================================================
    # Recipe
    # =================================================

    def recipe(
        self,
        result_name: str,
        *,
        preferred_only: bool = True,
    ) -> dict[str, Any]:
        self._ensure_open()

        result = self.store.get_recipe(
            result_name,
            preferred_only=(
                preferred_only
            ),
            include_record=False,
        )

        if result["status"] == "not_found":
            return {
                "status": "not_found",
                "intent": "recipe",
                "query": result_name,
                "facts": None,
                "candidates": [],
                "warnings": [
                    "No matching recipe result."
                ],
                "provenance": [],
            }

        if result["status"] == "ambiguous":
            candidates = [
                {
                    "name": recipe[
                        "result_name"
                    ],
                    "result_item_id": (
                        recipe[
                            "result_item_id"
                        ]
                    ),
                    "source_catalog_id": (
                        recipe[
                            "source_catalog_id"
                        ]
                    ),
                }
                for recipe
                in result["recipes"]
            ]

            return {
                "status": "ambiguous",
                "intent": "recipe",
                "query": result_name,
                "facts": None,
                "candidates": candidates,
                "warnings": [
                    "Multiple recipes share this "
                    "result name."
                ],
                "provenance": [
                    {
                        "entity_type": "recipe",
                        "source_catalog_id": (
                            candidate[
                                "source_catalog_id"
                            ]
                        ),
                    }
                    for candidate in candidates
                ],
            }

        recipe = result["recipe"]

        variants = []
        warnings = []
        provenance = [
            {
                "entity_type": "recipe",
                "source_catalog_id": (
                    recipe[
                        "source_catalog_id"
                    ]
                ),
            }
        ]

        if recipe["result_item_catalog_id"]:
            provenance.append(
                {
                    "entity_type": "item",
                    "source_catalog_id": (
                        recipe[
                            "result_item_catalog_id"
                        ]
                    ),
                }
            )

        for variant in recipe["variants"]:
            ingredients = []

            for ingredient in variant[
                "ingredients"
            ]:
                ingredient_fact = {
                    "name": ingredient["name"],
                    "quantity": (
                        ingredient["quantity"]
                    ),
                    "kind": ingredient["kind"],
                    "link_status": (
                        ingredient[
                            "link_status"
                        ]
                    ),
                    "item_id": ingredient.get(
                        "item_id"
                    ),
                    "group": ingredient.get(
                        "group"
                    ),
                }

                ingredients.append(
                    ingredient_fact
                )

                item_catalog_id = (
                    ingredient.get(
                        "item_catalog_id"
                    )
                )

                if item_catalog_id:
                    provenance.append(
                        {
                            "entity_type": "item",
                            "source_catalog_id": (
                                item_catalog_id
                            ),
                        }
                    )

                if (
                    ingredient[
                        "link_status"
                    ]
                    == "unresolved"
                ):
                    warnings.append(
                        "Unresolved recipe ingredient: "
                        f"{ingredient['name']}."
                    )

            variants.append(
                {
                    "variant_id": (
                        variant["variant_id"]
                    ),
                    "is_current": bool(
                        variant["is_current"]
                    ),
                    "is_legacy": bool(
                        variant["is_legacy"]
                    ),
                    "is_preferred": bool(
                        variant["is_preferred"]
                    ),
                    "version_label": (
                        variant[
                            "version_label"
                        ]
                    ),
                    "result_quantity": (
                        variant[
                            "result_quantity"
                        ]
                    ),
                    "stations": (
                        variant["stations"]
                    ),
                    "ingredients": ingredients,
                }
            )

        if recipe["linking_status"] == "partial":
            warnings.append(
                "The complete recipe record contains "
                "legacy or unresolved references."
            )

        facts = {
            "result_name": (
                recipe["result_name"]
            ),
            "result_item_id": (
                recipe["result_item_id"]
            ),
            "preferred_only": preferred_only,
            "variant_count": len(variants),
            "variants": variants,
            "linking_status": (
                recipe["linking_status"]
            ),
        }

        return {
            "status": "found",
            "intent": "recipe",
            "query": result_name,
            "facts": facts,
            "candidates": [],
            "warnings": sorted(
                set(warnings)
            ),
            "provenance": (
                self._unique_provenance(
                    provenance
                )
            ),
        }

    # =================================================
    # Recipes using Item
    # =================================================

    def recipes_using_item(
        self,
        item_name: str,
        *,
        preferred_only: bool = True,
    ) -> dict[str, Any]:
        self._ensure_open()

        result = self.store.recipes_using_item(
            item_name,
            preferred_only=(
                preferred_only
            ),
        )

        status = result["status"]

        if status == "item_not_found":
            return {
                "status": "not_found",
                "intent": (
                    "recipes_using_item"
                ),
                "query": item_name,
                "facts": None,
                "candidates": [],
                "warnings": [
                    "The ingredient Item was not found."
                ],
                "provenance": [],
            }

        if status == "item_ambiguous":
            candidates = [
                self._candidate_summary(
                    match,
                    entity_type="item",
                )
                for match
                in result["item_matches"]
            ]

            return {
                "status": "ambiguous",
                "intent": (
                    "recipes_using_item"
                ),
                "query": item_name,
                "facts": None,
                "candidates": candidates,
                "warnings": [
                    "The ingredient Item name is "
                    "ambiguous."
                ],
                "provenance": [
                    {
                        "entity_type": "item",
                        "source_catalog_id": (
                            candidate[
                                "source_catalog_id"
                            ]
                        ),
                    }
                    for candidate in candidates
                ],
            }

        item = result["item"]
        recipes = result["recipes"]

        provenance = [
            {
                "entity_type": "item",
                "source_catalog_id": (
                    item["source_catalog_id"]
                ),
            }
        ]

        for recipe in recipes:
            provenance.append(
                {
                    "entity_type": "recipe",
                    "source_catalog_id": (
                        recipe[
                            "source_catalog_id"
                        ]
                    ),
                }
            )

        return {
            "status": "found",
            "intent": "recipes_using_item",
            "query": item_name,
            "facts": {
                "item": {
                    "name": item["name"],
                    "item_id": item[
                        "item_id"
                    ],
                },
                "preferred_only": (
                    preferred_only
                ),
                "recipe_count": len(
                    recipes
                ),
                "recipes": [
                    {
                        "result_name": (
                            recipe[
                                "result_name"
                            ]
                        ),
                        "result_item_id": (
                            recipe[
                                "result_item_id"
                            ]
                        ),
                        "linking_status": (
                            recipe[
                                "linking_status"
                            ]
                        ),
                    }
                    for recipe in recipes
                ],
            },
            "candidates": [],
            "warnings": [],
            "provenance": (
                self._unique_provenance(
                    provenance
                )
            ),
        }

    # =================================================
    # Drops
    # =================================================

    def _format_drop(
        self,
        drop: dict[str, Any],
        *,
        mode: str,
    ) -> dict[str, Any]:
        return {
            "item_name": drop[
                "item_name"
            ],
            "item_id": drop.get(
                "item_id"
            ),
            "item_kind": drop[
                "item_kind"
            ],
            "source_name": drop[
                "source_name"
            ],
            "source_kind": drop[
                "source_kind"
            ],
            "npc_id": drop.get(
                "npc_id"
            ),
            "mode": mode,
            "chance": (
                self._format_probability(
                    drop.get("chance")
                )
            ),
            "quantity": (
                self._format_quantity(
                    drop.get(
                        "quantity_for_mode"
                    )
                )
            ),
            "quantity_by_condition": (
                drop.get(
                    "quantity_by_condition",
                    {},
                )
            ),
            "availability": drop[
                "availability"
            ],
            "conditions": drop[
                "conditions"
            ],
            "item_group": drop.get(
                "item_group"
            ),
            "source_group": drop.get(
                "source_group"
            ),
            "linking_status": drop[
                "linking_status"
            ],
            "parse_status": drop[
                "parse_status"
            ],
            "source_catalog_id": drop[
                "source_catalog_id"
            ],
            "item_catalog_id": drop.get(
                "item_catalog_id"
            ),
            "npc_catalog_id": drop.get(
                "npc_catalog_id"
            ),
        }

    def drops_for_item(
        self,
        item_name: str,
        *,
        mode: str = "normal",
        include_partial: bool = True,
    ) -> dict[str, Any]:
        self._ensure_open()

        result = self.store.drops_for_item(
            item_name,
            mode=mode,
            include_partial=(
                include_partial
            ),
            include_record=False,
        )

        if result["status"] == "item_not_found":
            return {
                "status": "not_found",
                "intent": "drops_for_item",
                "query": item_name,
                "facts": None,
                "candidates": [],
                "warnings": [
                    "No matching Item or Drop name."
                ],
                "provenance": [],
            }

        if result["status"] == "item_ambiguous":
            candidates = [
                self._candidate_summary(
                    match,
                    entity_type="item",
                )
                for match
                in result["item_matches"]
            ]

            return {
                "status": "ambiguous",
                "intent": "drops_for_item",
                "query": item_name,
                "facts": None,
                "candidates": candidates,
                "warnings": [
                    "The Item name is ambiguous."
                ],
                "provenance": [
                    {
                        "entity_type": "item",
                        "source_catalog_id": (
                            candidate[
                                "source_catalog_id"
                            ]
                        ),
                    }
                    for candidate in candidates
                ],
            }

        drops = [
            self._format_drop(
                drop,
                mode=mode,
            )
            for drop in result["drops"]
        ]

        warnings = []

        if any(
            drop["linking_status"]
            == "partial"
            for drop in drops
        ):
            warnings.append(
                "Some Drop records contain legacy "
                "or unresolved references."
            )

        provenance = []

        for drop in drops:
            provenance.append(
                {
                    "entity_type": "drop",
                    "source_catalog_id": (
                        drop[
                            "source_catalog_id"
                        ]
                    ),
                }
            )

            if drop["item_catalog_id"]:
                provenance.append(
                    {
                        "entity_type": "item",
                        "source_catalog_id": (
                            drop[
                                "item_catalog_id"
                            ]
                        ),
                    }
                )

            if drop["npc_catalog_id"]:
                provenance.append(
                    {
                        "entity_type": "npc",
                        "source_catalog_id": (
                            drop[
                                "npc_catalog_id"
                            ]
                        ),
                    }
                )

        return {
            "status": "found",
            "intent": "drops_for_item",
            "query": item_name,
            "facts": {
                "mode": mode,
                "include_partial": (
                    include_partial
                ),
                "drop_count": len(drops),
                "drops": drops,
            },
            "candidates": [],
            "warnings": warnings,
            "provenance": (
                self._unique_provenance(
                    provenance
                )
            ),
        }

    def drops_from_source(
        self,
        source_name: str,
        *,
        mode: str = "normal",
        include_partial: bool = True,
    ) -> dict[str, Any]:
        self._ensure_open()

        result = self.store.drops_from_source(
            source_name,
            mode=mode,
            include_partial=(
                include_partial
            ),
            include_record=False,
        )

        if result["status"] == "not_found":
            return {
                "status": "not_found",
                "intent": "drops_from_source",
                "query": source_name,
                "facts": None,
                "candidates": [],
                "warnings": [
                    "No matching Drop source."
                ],
                "provenance": [],
            }

        drops = [
            self._format_drop(
                drop,
                mode=mode,
            )
            for drop in result["drops"]
        ]

        warnings = []

        if any(
            drop["linking_status"]
            == "partial"
            for drop in drops
        ):
            warnings.append(
                "Some Drop records contain legacy "
                "or unresolved references."
            )

        provenance = []

        for drop in drops:
            provenance.append(
                {
                    "entity_type": "drop",
                    "source_catalog_id": (
                        drop[
                            "source_catalog_id"
                        ]
                    ),
                }
            )

            if drop["item_catalog_id"]:
                provenance.append(
                    {
                        "entity_type": "item",
                        "source_catalog_id": (
                            drop[
                                "item_catalog_id"
                            ]
                        ),
                    }
                )

            if drop["npc_catalog_id"]:
                provenance.append(
                    {
                        "entity_type": "npc",
                        "source_catalog_id": (
                            drop[
                                "npc_catalog_id"
                            ]
                        ),
                    }
                )

        return {
            "status": "found",
            "intent": "drops_from_source",
            "query": source_name,
            "facts": {
                "mode": mode,
                "include_partial": (
                    include_partial
                ),
                "source_kinds": (
                    result["source_kinds"]
                ),
                "drop_count": len(drops),
                "drops": drops,
            },
            "candidates": [],
            "warnings": warnings,
            "provenance": (
                self._unique_provenance(
                    provenance
                )
            ),
        }

    # =================================================
    # Search and router
    # =================================================

    def search(
        self,
        query: str,
        *,
        limit_per_type: int = 10,
    ) -> dict[str, Any]:
        self._ensure_open()

        result = self.store.search(
            query,
            limit_per_type=(
                limit_per_type
            ),
        )

        return {
            "status": "found",
            "intent": "search",
            "query": query,
            "facts": result,
            "candidates": [],
            "warnings": [],
            "provenance": (
                self._unique_provenance(
                    [
                        {
                            "entity_type": "item",
                            "source_catalog_id": (
                                item[
                                    "source_catalog_id"
                                ]
                            ),
                        }
                        for item
                        in result["items"]
                    ]
                    + [
                        {
                            "entity_type": "npc",
                            "source_catalog_id": (
                                npc[
                                    "source_catalog_id"
                                ]
                            ),
                        }
                        for npc
                        in result["npcs"]
                    ]
                    + [
                        {
                            "entity_type": "recipe",
                            "source_catalog_id": (
                                recipe[
                                    "source_catalog_id"
                                ]
                            ),
                        }
                        for recipe
                        in result["recipes"]
                    ]
                )
            ),
        }

    def query(
        self,
        intent: str,
        entity: str,
        *,
        mode: str = "normal",
        npc_id: int | None = None,
        preferred_only: bool = True,
        include_partial: bool = True,
        limit_per_type: int = 10,
    ) -> dict[str, Any]:
        self._ensure_open()

        normalized_intent = str(
            intent
        ).strip().casefold()

        if normalized_intent not in (
            self.VALID_INTENTS
        ):
            raise ValueError(
                "Unsupported intent. Expected one of: "
                + ", ".join(
                    sorted(
                        self.VALID_INTENTS
                    )
                )
            )

        if normalized_intent == "item":
            return self.item(entity)

        if normalized_intent == "npc":
            return self.npc(
                entity,
                npc_id=npc_id,
            )

        if normalized_intent == "recipe":
            return self.recipe(
                entity,
                preferred_only=(
                    preferred_only
                ),
            )

        if (
            normalized_intent
            == "recipes_using_item"
        ):
            return self.recipes_using_item(
                entity,
                preferred_only=(
                    preferred_only
                ),
            )

        if (
            normalized_intent
            == "drops_for_item"
        ):
            return self.drops_for_item(
                entity,
                mode=mode,
                include_partial=(
                    include_partial
                ),
            )

        if (
            normalized_intent
            == "drops_from_source"
        ):
            return self.drops_from_source(
                entity,
                mode=mode,
                include_partial=(
                    include_partial
                ),
            )

        return self.search(
            entity,
            limit_per_type=(
                limit_per_type
            ),
        )

    def to_context_json(
        self,
        result: dict[str, Any],
        *,
        indent: int | None = 2,
    ) -> str:
        return json.dumps(
            result,
            ensure_ascii=False,
            indent=indent,
        )
