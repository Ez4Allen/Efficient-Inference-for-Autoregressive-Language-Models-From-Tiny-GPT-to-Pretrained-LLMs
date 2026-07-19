
from __future__ import annotations

from typing import Any


COMMON_REQUIRED_FIELDS = {
    "id",
    "fact_type",
    "entity_name",
}

SUPPORTED_FACT_TYPES = {
    "recipe",
    "boss_summon",
}


def validate_record(
    record: dict[str, Any],
) -> None:
    """
    Validate one structured Terraria fact.

    The function returns None when the record is valid.
    It raises ValueError or TypeError when the record is invalid.
    """
    if not isinstance(record, dict):
        raise TypeError(
            "Knowledge record must be a dictionary."
        )

    missing_fields = (
        COMMON_REQUIRED_FIELDS - record.keys()
    )

    if missing_fields:
        raise ValueError(
            "Record is missing required fields: "
            f"{sorted(missing_fields)}"
        )

    _validate_non_empty_string(
        record["id"],
        field_name="id",
    )

    _validate_non_empty_string(
        record["fact_type"],
        field_name="fact_type",
    )

    _validate_non_empty_string(
        record["entity_name"],
        field_name="entity_name",
    )

    fact_type = record["fact_type"]

    if fact_type not in SUPPORTED_FACT_TYPES:
        raise ValueError(
            f"Unsupported fact_type: {fact_type!r}. "
            f"Supported types: "
            f"{sorted(SUPPORTED_FACT_TYPES)}"
        )

    aliases = record.get("aliases", [])

    _validate_string_list(
        aliases,
        field_name="aliases",
    )

    if "platforms" in record:
        _validate_string_list(
            record["platforms"],
            field_name="platforms",
        )

    if "notes" in record:
        _validate_string_list(
            record["notes"],
            field_name="notes",
        )

    if "verified" in record:
        if not isinstance(record["verified"], bool):
            raise TypeError(
                "Field 'verified' must be a boolean."
            )

    if fact_type == "recipe":
        _validate_recipe(record)

    elif fact_type == "boss_summon":
        _validate_boss_summon(record)



def _validate_recipe(
    record: dict[str, Any],
) -> None:
    """
    Validate a recipe record.

    Two formats are supported:

    Legacy format:
        ingredients
        crafting_stations
        conditions

    Preferred format:
        recipe_variants
    """
    if "recipe_variants" in record:
        _validate_recipe_variants(
            record["recipe_variants"]
        )

        return

    # Backward-compatible validation for older records.
    required_fields = {
        "ingredients",
        "crafting_stations",
    }

    missing_fields = (
        required_fields - record.keys()
    )

    if missing_fields:
        raise ValueError(
            "Recipe record must contain either "
            "'recipe_variants' or the legacy fields: "
            f"{sorted(missing_fields)}"
        )

    _validate_ingredients(
        record["ingredients"],
        field_name="ingredients",
    )

    _validate_string_list(
        record["crafting_stations"],
        field_name="crafting_stations",
        allow_empty=False,
    )

    if "conditions" in record:
        _validate_string_list(
            record["conditions"],
            field_name="conditions",
        )


def _validate_recipe_variants(
    variants: Any,
) -> None:
    """
    Validate all alternative recipes for one entity.
    """
    if not isinstance(variants, list):
        raise TypeError(
            "Recipe field 'recipe_variants' "
            "must be a list."
        )

    if not variants:
        raise ValueError(
            "Recipe field 'recipe_variants' "
            "cannot be empty."
        )

    seen_variant_ids: set[str] = set()

    for variant_index, variant in enumerate(
        variants
    ):
        field_prefix = (
            f"recipe_variants[{variant_index}]"
        )

        if not isinstance(variant, dict):
            raise TypeError(
                f"{field_prefix} must be a dictionary."
            )

        required_fields = {
            "ingredients",
            "crafting_stations",
        }

        missing_fields = (
            required_fields - variant.keys()
        )

        if missing_fields:
            raise ValueError(
                f"{field_prefix} is missing fields: "
                f"{sorted(missing_fields)}"
            )

        variant_id = variant.get("variant_id")

        if variant_id is not None:
            _validate_non_empty_string(
                variant_id,
                field_name=(
                    f"{field_prefix}.variant_id"
                ),
            )

            if variant_id in seen_variant_ids:
                raise ValueError(
                    "Duplicate recipe variant id: "
                    f"{variant_id!r}"
                )

            seen_variant_ids.add(variant_id)

        if "label" in variant:
            _validate_non_empty_string(
                variant["label"],
                field_name=f"{field_prefix}.label",
            )

        _validate_ingredients(
            variant["ingredients"],
            field_name=(
                f"{field_prefix}.ingredients"
            ),
        )

        _validate_string_list(
            variant["crafting_stations"],
            field_name=(
                f"{field_prefix}.crafting_stations"
            ),
            allow_empty=False,
        )

        if "conditions" in variant:
            _validate_string_list(
                variant["conditions"],
                field_name=(
                    f"{field_prefix}.conditions"
                ),
            )

        if "notes" in variant:
            _validate_string_list(
                variant["notes"],
                field_name=(
                    f"{field_prefix}.notes"
                ),
            )


def _validate_ingredients(
    ingredients: Any,
    field_name: str,
) -> None:
    """
    Validate one ingredient list.
    """
    if not isinstance(ingredients, list):
        raise TypeError(
            f"Field {field_name!r} must be a list."
        )

    if not ingredients:
        raise ValueError(
            f"Field {field_name!r} cannot be empty."
        )

    for ingredient_index, ingredient in enumerate(
        ingredients
    ):
        ingredient_field = (
            f"{field_name}[{ingredient_index}]"
        )

        if not isinstance(ingredient, dict):
            raise TypeError(
                f"{ingredient_field} "
                "must be a dictionary."
            )

        if "item" not in ingredient:
            raise ValueError(
                f"{ingredient_field} "
                "is missing 'item'."
            )

        _validate_non_empty_string(
            ingredient["item"],
            field_name=(
                f"{ingredient_field}.item"
            ),
        )

        if "quantity" not in ingredient:
            raise ValueError(
                f"{ingredient_field} "
                "is missing 'quantity'."
            )

        quantity = ingredient["quantity"]

        # bool is a subclass of int in Python,
        # but True/False should not be valid quantities.
        if (
            isinstance(quantity, bool)
            or not isinstance(quantity, (int, float))
        ):
            raise TypeError(
                f"{ingredient_field}.quantity "
                "must be a number."
            )

        if quantity <= 0:
            raise ValueError(
                f"{ingredient_field}.quantity "
                "must be greater than zero."
            )


def _validate_boss_summon(
    record: dict[str, Any],
) -> None:
    required_fields = {
        "summons",
        "usage_conditions",
        "biome_requirements",
        "failure_conditions",
    }

    missing_fields = (
        required_fields - record.keys()
    )

    if missing_fields:
        raise ValueError(
            "Boss summon record is missing fields: "
            f"{sorted(missing_fields)}"
        )

    _validate_non_empty_string(
        record["summons"],
        field_name="summons",
    )

    _validate_string_list(
        record["usage_conditions"],
        field_name="usage_conditions",
    )

    _validate_string_list(
        record["biome_requirements"],
        field_name="biome_requirements",
    )

    _validate_string_list(
        record["failure_conditions"],
        field_name="failure_conditions",
    )


def _validate_non_empty_string(
    value: Any,
    field_name: str,
) -> None:
    if not isinstance(value, str):
        raise TypeError(
            f"Field {field_name!r} must be a string."
        )

    if not value.strip():
        raise ValueError(
            f"Field {field_name!r} cannot be empty."
        )


def _validate_string_list(
    value: Any,
    field_name: str,
    allow_empty: bool = True,
) -> None:
    if not isinstance(value, list):
        raise TypeError(
            f"Field {field_name!r} must be a list."
        )

    if not allow_empty and not value:
        raise ValueError(
            f"Field {field_name!r} cannot be empty."
        )

    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise TypeError(
                f"{field_name}[{index}] must be a string."
            )

        if not item.strip():
            raise ValueError(
                f"{field_name}[{index}] cannot be empty."
            )
