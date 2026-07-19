
from __future__ import annotations

from typing import Any


def render_query_result(
    result: dict[str, Any],
) -> str:
    """
    Convert a structured query result into deterministic text.

    This renderer does not use an LLM.
    It only formats facts returned by the structured database.
    """
    route = result.get("route")

    if route == "fallback":
        query = result.get("query", "the requested entity")

        return (
            f"No exact structured fact was found for "
            f"{query!r}."
        )

    if route == "clarify":
        candidates = result.get("candidates", [])

        if not candidates:
            return (
                "Multiple records matched, but no candidate "
                "information is available."
            )

        lines = [
            "Multiple records matched. Please specify one:",
        ]

        for candidate in candidates:
            entity = candidate.get("entity", "Unknown")
            fact_type = candidate.get(
                "fact_type",
                "unknown",
            )

            lines.append(
                f"- {entity} ({fact_type})"
            )

        return "\n".join(lines)

    if route != "structured_fact":
        return "Unsupported query result."

    entity = result["entity"]
    fact_type = result["fact_type"]
    facts = result.get("facts") or {}

    if fact_type == "recipe":
        return _render_recipe(
            entity=entity,
            facts=facts,
        )

    if fact_type == "boss_summon":
        return _render_boss_summon(
            entity=entity,
            facts=facts,
        )

    return _render_generic_fact(
        entity=entity,
        fact_type=fact_type,
        facts=facts,
    )



def _render_recipe(
    entity: str,
    facts: dict[str, Any],
) -> str:
    """
    Render either a legacy recipe or multiple
    alternative recipe variants.
    """
    schema = facts.get(
        "schema",
        "legacy_recipe",
    )

    if schema == "recipe_variants":
        return _render_recipe_variants(
            entity=entity,
            facts=facts,
        )

    return _render_legacy_recipe(
        entity=entity,
        facts=facts,
    )


def _render_legacy_recipe(
    entity: str,
    facts: dict[str, Any],
) -> str:
    """
    Render the original single-recipe schema.
    """
    lines = [
        f"{entity} recipe:",
    ]

    _append_ingredients(
        lines=lines,
        ingredients=facts.get(
            "ingredients",
            [],
        ),
    )

    _append_crafting_stations(
        lines=lines,
        crafting_stations=facts.get(
            "crafting_stations",
            [],
        ),
    )

    _append_string_section(
        lines=lines,
        title="Conditions",
        values=facts.get(
            "conditions",
            [],
        ),
    )

    _append_string_section(
        lines=lines,
        title="Notes",
        values=facts.get(
            "notes",
            [],
        ),
    )

    return "\n".join(lines)


def _render_recipe_variants(
    entity: str,
    facts: dict[str, Any],
) -> str:
    """
    Render all alternative recipes for one entity.
    """
    variants = facts.get(
        "recipe_variants",
        [],
    )

    if not variants:
        return (
            f"No recipe variants are recorded for "
            f"{entity}."
        )

    lines = [
        f"{entity} recipes:",
    ]

    for index, variant in enumerate(
        variants,
        start=1,
    ):
        label = variant.get(
            "label",
            f"Recipe {index}",
        )

        lines.append("")
        lines.append(
            f"Option {index}: {label}"
        )

        _append_ingredients(
            lines=lines,
            ingredients=variant.get(
                "ingredients",
                [],
            ),
        )

        _append_crafting_stations(
            lines=lines,
            crafting_stations=variant.get(
                "crafting_stations",
                [],
            ),
        )

        _append_string_section(
            lines=lines,
            title="Conditions",
            values=variant.get(
                "conditions",
                [],
            ),
        )

        _append_string_section(
            lines=lines,
            title="Variant notes",
            values=variant.get(
                "notes",
                [],
            ),
        )

    _append_string_section(
        lines=lines,
        title="General notes",
        values=facts.get(
            "notes",
            [],
        ),
    )

    return "\n".join(lines)


def _append_ingredients(
    lines: list[str],
    ingredients: list[dict[str, Any]],
) -> None:
    """
    Add one ingredient section to an output line list.
    """
    if not ingredients:
        return

    lines.append("")
    lines.append("Ingredients:")

    for ingredient in ingredients:
        item = ingredient.get(
            "item",
            "Unknown item",
        )

        quantity = ingredient.get(
            "quantity",
        )

        if quantity is None:
            lines.append(f"- {item}")
        else:
            lines.append(
                f"- {quantity} x {item}"
            )


def _append_crafting_stations(
    lines: list[str],
    crafting_stations: list[str],
) -> None:
    """
    Add crafting stations to an output line list.
    """
    if not crafting_stations:
        return

    title = (
        "Crafting station"
        if len(crafting_stations) == 1
        else "Crafting stations"
    )

    lines.append("")
    lines.append(
        f"{title}: "
        + " or ".join(crafting_stations)
    )


def _append_string_section(
    lines: list[str],
    title: str,
    values: list[str],
) -> None:
    """
    Add a titled bullet-list section when values exist.
    """
    if not values:
        return

    lines.append("")
    lines.append(f"{title}:")

    for value in values:
        lines.append(f"- {value}")


def _render_boss_summon(
    entity: str,
    facts: dict[str, Any],
) -> str:
    summons = facts.get("summons")

    if summons:
        lines = [
            f"{entity} summons {summons}.",
        ]
    else:
        lines = [
            f"No summon target is recorded for {entity}.",
        ]

    usage_conditions = facts.get(
        "usage_conditions",
        [],
    )

    if usage_conditions:
        lines.append("")
        lines.append("Usage conditions:")

        for condition in usage_conditions:
            lines.append(f"- {condition}")

    biome_requirements = facts.get(
        "biome_requirements",
        [],
    )

    lines.append("")
    lines.append("Biome requirements:")

    if biome_requirements:
        for requirement in biome_requirements:
            lines.append(f"- {requirement}")
    else:
        lines.append("- None")

    failure_conditions = facts.get(
        "failure_conditions",
        [],
    )

    if failure_conditions:
        lines.append("")
        lines.append("Failure conditions:")

        for condition in failure_conditions:
            lines.append(f"- {condition}")

    notes = facts.get(
        "notes",
        [],
    )

    if notes:
        lines.append("")
        lines.append("Notes:")

        for note in notes:
            lines.append(f"- {note}")

    return "\n".join(lines)


def _render_generic_fact(
    entity: str,
    fact_type: str,
    facts: dict[str, Any],
) -> str:
    lines = [
        f"Entity: {entity}",
        f"Fact type: {fact_type}",
    ]

    for key, value in facts.items():
        readable_key = key.replace(
            "_",
            " ",
        ).title()

        lines.append(
            f"{readable_key}: {value}"
        )

    return "\n".join(lines)
