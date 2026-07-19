
from __future__ import annotations

import json
import re

from collections import Counter
from pathlib import Path
from typing import Any

from ..catalog_store import normalize_catalog_name
from .common import (
    normalize_whitespace,
    parse_bool,
    parse_int,
    strip_markup,
)


DEFAULT_INPUT_PATH = Path(
    "/content/llm_project/data/terraria/"
    "catalog/normalized/recipes_grouped.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "/content/llm_project/data/terraria/"
    "catalog/cleaned/Recipes.jsonl"
)

DEFAULT_REPORT_PATH = Path(
    "/content/llm_project/data/terraria/"
    "catalog/cleaned/Recipes_report.json"
)


def _is_empty(
    value: Any,
) -> bool:
    return (
        value is None
        or not str(value).strip()
    )


def _clean_name(
    value: Any,
) -> str | None:
    cleaned = strip_markup(
        value
    )

    if not cleaned:
        return None

    return normalize_whitespace(
        cleaned
    )


def _unique_strings(
    values: list[Any],
) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()

    for value in values:
        cleaned = _clean_name(
            value
        )

        if not cleaned:
            continue

        deduplication_key = (
            cleaned.casefold()
        )

        if deduplication_key in seen:
            continue

        seen.add(
            deduplication_key
        )

        output.append(
            cleaned
        )

    return output


def _normalize_version_key(
    value: Any,
) -> str | None:
    cleaned = _clean_name(
        value
    )

    if not cleaned:
        return None

    return re.sub(
        r"[^a-z0-9]+",
        "_",
        cleaned.casefold(),
    ).strip("_")


def _clean_version(
    value: Any,
) -> dict[str, str | None]:
    label = _clean_name(
        value
    )

    return {
        "label": label,
        "normalized": (
            _normalize_version_key(
                label
            )
        ),
    }


def _clean_ingredient(
    ingredient: Any,
    *,
    variant_id: str,
    ingredient_index: int,
    warnings: list[str],
) -> dict[str, Any] | None:
    if not isinstance(
        ingredient,
        dict,
    ):
        warnings.append(
            "invalid_ingredient_record"
        )

        return None

    item_name = _clean_name(
        ingredient.get("item")
    )

    quantity = parse_int(
        ingredient.get("quantity")
    )

    if not item_name:
        warnings.append(
            "missing_ingredient_name"
        )

        return None

    if (
        quantity is None
        or quantity <= 0
    ):
        warnings.append(
            "invalid_ingredient_quantity"
        )

        return None

    return {
        "position": ingredient_index,

        "item": {
            "name": item_name,
            "normalized_name": (
                normalize_catalog_name(
                    item_name
                )
            ),
            "item_id": None,
        },

        "quantity": quantity,

        "source": {
            "variant_id": variant_id,
        },
    }


def _clean_variant(
    variant: Any,
    *,
    variant_index: int,
) -> dict[str, Any]:
    warnings: list[str] = []

    if not isinstance(
        variant,
        dict,
    ):
        return {
            "variant_id": (
                f"invalid_variant_"
                f"{variant_index}"
            ),
            "position": variant_index,
            "result_quantity": None,
            "ingredients": [],
            "crafting_stations": [],
            "version": {
                "label": None,
                "normalized": None,
            },
            "is_legacy": False,
            "is_current": True,
            "note": None,
            "source": {},
            "parse_status": "partial",
            "parse_warnings": [
                "invalid_variant_record",
            ],
        }

    variant_id = _clean_name(
        variant.get("variant_id")
    )

    if not variant_id:
        variant_id = (
            f"missing_variant_"
            f"{variant_index}"
        )

        warnings.append(
            "missing_variant_id"
        )

    result_quantity = parse_int(
        variant.get("result_quantity")
    )

    if (
        result_quantity is None
        or result_quantity <= 0
    ):
        warnings.append(
            "invalid_result_quantity"
        )

    raw_ingredients = (
        variant.get("ingredients")
    )

    ingredients: list[
        dict[str, Any]
    ] = []

    if isinstance(
        raw_ingredients,
        list,
    ):
        for ingredient_index, ingredient in enumerate(
            raw_ingredients,
            start=1,
        ):
            cleaned_ingredient = (
                _clean_ingredient(
                    ingredient,
                    variant_id=variant_id,
                    ingredient_index=(
                        ingredient_index
                    ),
                    warnings=warnings,
                )
            )

            if cleaned_ingredient:
                ingredients.append(
                    cleaned_ingredient
                )

    else:
        warnings.append(
            "invalid_ingredients_list"
        )

    if not ingredients:
        warnings.append(
            "missing_ingredients"
        )

    raw_stations = variant.get(
        "crafting_stations"
    )

    if isinstance(raw_stations, list):
        crafting_stations = (
            _unique_strings(
                raw_stations
            )
        )
    else:
        crafting_stations = []

        warnings.append(
            "invalid_crafting_stations"
        )

    if not crafting_stations:
        warnings.append(
            "missing_crafting_station"
        )

    legacy_value = parse_bool(
        variant.get("legacy"),
        empty_value=False,
    )

    is_legacy = bool(
        legacy_value
    )

    raw_source = variant.get("raw")

    if not isinstance(
        raw_source,
        dict,
    ):
        raw_source = {}

    note = _clean_name(
        variant.get("note")
    )

    return {
        "variant_id": variant_id,
        "position": variant_index,

        "result_quantity": (
            result_quantity
        ),

        "ingredients": ingredients,

        "crafting_stations": [
            {
                "name": station_name,
                "normalized_name": (
                    normalize_catalog_name(
                        station_name
                    )
                ),
            }
            for station_name
            in crafting_stations
        ],

        "version": _clean_version(
            variant.get("version")
        ),

        "is_legacy": is_legacy,
        "is_current": not is_legacy,

        "note": note,

        "source": {
            "row_id": raw_source.get(
                "_rowID"
            ),
            "page_name": raw_source.get(
                "_pageName"
            ),
            "page_id": raw_source.get(
                "_pageID"
            ),
            "page_namespace": (
                raw_source.get(
                    "_pageNamespace"
                )
            ),
            "result_quantity_raw": (
                variant.get(
                    "result_quantity_raw"
                )
            ),
            "legacy_raw": variant.get(
                "legacy"
            ),
            "version_raw": variant.get(
                "version"
            ),
        },

        "parse_status": (
            "ok"
            if not warnings
            else "partial"
        ),

        "parse_warnings": warnings,
    }



def _parse_result_identifier(
    value: Any,
) -> dict[str, Any]:
    """
    Preserve canonical, multi-version, and special
    Terraria result identifiers without guessing which
    candidate should be treated as the canonical ID.
    """
    raw_value = value

    if value is None or not str(value).strip():
        return {
            "item_id": None,
            "item_id_candidates": [],
            "raw_item_id": raw_value,
            "item_id_status": "missing",
            "has_item_id_reference": False,
        }

    raw_text = normalize_whitespace(
        str(value)
    )

    # Standard single Terraria item ID.
    if re.fullmatch(
        r"[+-]?\d+",
        raw_text,
    ):
        item_id = int(raw_text)

        return {
            "item_id": item_id,
            "item_id_candidates": [
                item_id,
            ],
            "raw_item_id": raw_value,
            "item_id_status": "canonical",
            "has_item_id_reference": True,
        }

    candidates: list[int] = []

    for number_text in re.findall(
        r"\d+",
        raw_text,
    ):
        candidate = int(
            number_text
        )

        if candidate not in candidates:
            candidates.append(
                candidate
            )

    # Example:
    #   5004 / 1870
    if re.fullmatch(
        r"\d+"
        r"(?:\s*/\s*\d+)+",
        raw_text,
    ):
        status = "multiple"

    # Example:
    #   5035sp
    elif candidates:
        status = "special"

    else:
        status = "unparsed"

    return {
        # Only a pure integer is exposed as canonical.
        "item_id": None,

        # Candidate numeric components remain available
        # for later platform/version mapping.
        "item_id_candidates": candidates,

        "raw_item_id": raw_value,
        "item_id_status": status,

        "has_item_id_reference": bool(
            candidates
        ),
    }

def clean_recipe_record(
    record: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise TypeError(
            "Recipe record must be "
            "a dictionary."
        )

    warnings: list[str] = []

    result_name = _clean_name(
        record.get("entity_name")
    )

    if not result_name:
        result_name = "Unknown Result"

        warnings.append(
            "missing_result_name"
        )

    result_identifier = (
        _parse_result_identifier(
            record.get("entity_id")
        )
    )

    result_item_id = (
        result_identifier["item_id"]
    )

    raw_variants = record.get(
        "recipe_variants"
    )

    variants: list[
        dict[str, Any]
    ] = []

    if isinstance(raw_variants, list):
        for variant_index, variant in enumerate(
            raw_variants,
            start=1,
        ):
            variants.append(
                _clean_variant(
                    variant,
                    variant_index=(
                        variant_index
                    ),
                )
            )
    else:
        warnings.append(
            "invalid_recipe_variants"
        )

    if not variants:
        warnings.append(
            "missing_recipe_variants"
        )

    current_variant_ids = [
        variant["variant_id"]
        for variant in variants
        if variant["is_current"]
    ]

    legacy_variant_ids = [
        variant["variant_id"]
        for variant in variants
        if variant["is_legacy"]
    ]

    # Current variants are preferred for ordinary
    # queries. If a result only has legacy recipes,
    # preserve usability by falling back to all
    # available variants.
    preferred_variant_ids = (
        current_variant_ids
        if current_variant_ids
        else [
            variant["variant_id"]
            for variant in variants
        ]
    )

    variant_warnings = [
        warning
        for variant in variants
        for warning in variant[
            "parse_warnings"
        ]
    ]

    all_warnings = (
        warnings
        + variant_warnings
    )

    return {
        "source_catalog_id": record.get(
            "catalog_id"
        ),

        "record_type": "recipe",

        "result": {
            "name": result_name,
            "normalized_name": (
                normalize_catalog_name(
                    result_name
                )
            ),
            # Canonical ID is populated only when the
            # source contains one unambiguous integer.
            "item_id": result_item_id,

            "has_item_id": (
                result_item_id is not None
            ),

            "item_id_candidates": (
                result_identifier[
                    "item_id_candidates"
                ]
            ),

            "raw_item_id": (
                result_identifier[
                    "raw_item_id"
                ]
            ),

            "item_id_status": (
                result_identifier[
                    "item_id_status"
                ]
            ),

            "has_item_id_reference": (
                result_identifier[
                    "has_item_id_reference"
                ]
            ),
        },

        "variants": variants,

        "variant_selection": {
            "preferred_variant_ids": (
                preferred_variant_ids
            ),
            "current_variant_ids": (
                current_variant_ids
            ),
            "legacy_variant_ids": (
                legacy_variant_ids
            ),
            "has_current_recipe": bool(
                current_variant_ids
            ),
            "has_legacy_recipe": bool(
                legacy_variant_ids
            ),
        },

        "statistics": {
            "variant_count": len(
                variants
            ),
            "current_variant_count": len(
                current_variant_ids
            ),
            "legacy_variant_count": len(
                legacy_variant_ids
            ),
            "preferred_variant_count": len(
                preferred_variant_ids
            ),
            "ingredient_entry_count": sum(
                len(
                    variant["ingredients"]
                )
                for variant in variants
            ),
        },

        "parse_status": (
            "ok"
            if not all_warnings
            else "partial"
        ),

        "parse_warnings": (
            all_warnings
        ),
    }


def clean_recipes_file(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    report_path = Path(report_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Recipes input file not found: "
            f"{input_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_output_path = (
        output_path.with_suffix(
            output_path.suffix + ".tmp"
        )
    )

    total_records = 0
    ok_records = 0
    partial_records = 0

    total_variants = 0
    current_variants = 0
    legacy_variants = 0
    total_ingredient_entries = 0

    records_with_current_recipe = 0
    records_with_legacy_recipe = 0
    mixed_current_legacy_records = 0
    legacy_only_records = 0
    missing_result_item_id_records = 0
    note_variant_records = 0
    versioned_variants = 0

    warning_counts: Counter[str] = (
        Counter()
    )

    result_id_status_counts: Counter[str] = (
        Counter()
    )

    seen_catalog_ids: set[str] = set()
    seen_variant_ids: set[str] = set()

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as input_file, temporary_output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        for line_number, line in enumerate(
            input_file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at "
                    f"{input_path}:{line_number}"
                ) from error

            cleaned = clean_recipe_record(
                record
            )

            source_catalog_id = cleaned.get(
                "source_catalog_id"
            )

            if not source_catalog_id:
                raise ValueError(
                    "Missing source_catalog_id at "
                    f"{input_path}:{line_number}"
                )

            if source_catalog_id in seen_catalog_ids:
                raise ValueError(
                    "Duplicate source_catalog_id: "
                    f"{source_catalog_id}"
                )

            seen_catalog_ids.add(
                source_catalog_id
            )

            for variant in cleaned["variants"]:
                variant_id = variant[
                    "variant_id"
                ]

                if variant_id in seen_variant_ids:
                    raise ValueError(
                        "Duplicate variant_id: "
                        f"{variant_id}"
                    )

                seen_variant_ids.add(
                    variant_id
                )

                if variant["note"]:
                    note_variant_records += 1

                if variant[
                    "version"
                ]["label"]:
                    versioned_variants += 1

            total_records += 1

            if cleaned[
                "parse_status"
            ] == "ok":
                ok_records += 1
            else:
                partial_records += 1

            statistics = cleaned[
                "statistics"
            ]

            total_variants += statistics[
                "variant_count"
            ]

            current_variants += statistics[
                "current_variant_count"
            ]

            legacy_variants += statistics[
                "legacy_variant_count"
            ]

            total_ingredient_entries += (
                statistics[
                    "ingredient_entry_count"
                ]
            )

            selection = cleaned[
                "variant_selection"
            ]

            if selection[
                "has_current_recipe"
            ]:
                records_with_current_recipe += 1

            if selection[
                "has_legacy_recipe"
            ]:
                records_with_legacy_recipe += 1

            if (
                selection[
                    "has_current_recipe"
                ]
                and selection[
                    "has_legacy_recipe"
                ]
            ):
                mixed_current_legacy_records += 1

            if (
                not selection[
                    "has_current_recipe"
                ]
                and selection[
                    "has_legacy_recipe"
                ]
            ):
                legacy_only_records += 1

            result_id_status = cleaned[
                "result"
            ]["item_id_status"]

            result_id_status_counts.update(
                [
                    result_id_status,
                ]
            )

            if not cleaned[
                "result"
            ]["has_item_id_reference"]:
                missing_result_item_id_records += 1

            warning_counts.update(
                cleaned["parse_warnings"]
            )

            output_file.write(
                json.dumps(
                    cleaned,
                    ensure_ascii=False,
                )
                + "\n"
            )

    if total_records == 0:
        temporary_output_path.unlink(
            missing_ok=True
        )

        raise ValueError(
            "Recipes input contained "
            "no records."
        )

    temporary_output_path.replace(
        output_path
    )

    report = {
        "input_path": str(input_path),
        "output_path": str(output_path),

        "total_records": total_records,
        "ok_records": ok_records,
        "partial_records": partial_records,

        "unique_source_catalog_ids": len(
            seen_catalog_ids
        ),

        "total_variants": total_variants,
        "unique_variant_ids": len(
            seen_variant_ids
        ),

        "current_variants": (
            current_variants
        ),
        "legacy_variants": (
            legacy_variants
        ),

        "total_ingredient_entries": (
            total_ingredient_entries
        ),

        "records_with_current_recipe": (
            records_with_current_recipe
        ),

        "records_with_legacy_recipe": (
            records_with_legacy_recipe
        ),

        "mixed_current_legacy_records": (
            mixed_current_legacy_records
        ),

        "legacy_only_records": (
            legacy_only_records
        ),

        "missing_result_item_id_records": (
            missing_result_item_id_records
        ),

        "result_id_status_counts": dict(
            result_id_status_counts.most_common()
        ),

        "versioned_variants": (
            versioned_variants
        ),

        "note_variant_records": (
            note_variant_records
        ),

        "warning_counts": dict(
            warning_counts.most_common()
        ),
    }

    temporary_report_path = (
        report_path.with_suffix(
            report_path.suffix + ".tmp"
        )
    )

    temporary_report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary_report_path.replace(
        report_path
    )

    return report
