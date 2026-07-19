
from __future__ import annotations

import copy
import json

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_ITEMS_PATH = Path(
    "/content/llm_project/data/terraria/"
    "catalog/cleaned/Items.jsonl"
)

DEFAULT_RECIPES_PATH = Path(
    "/content/llm_project/data/terraria/"
    "catalog/cleaned/Recipes.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "/content/llm_project/data/terraria/"
    "catalog/linked/Recipes.jsonl"
)

DEFAULT_REPORT_PATH = Path(
    "/content/llm_project/data/terraria/"
    "catalog/linked/Recipes_link_report.json"
)


def _load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"JSONL file not found: {path}"
        )

    records: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
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
                    f"{path}:{line_number}"
                ) from error

            if not isinstance(record, dict):
                raise TypeError(
                    f"Record at {path}:{line_number} "
                    "is not a dictionary."
                )

            records.append(record)

    return records


def _item_reference(
    item: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source_catalog_id": item.get(
            "source_catalog_id"
        ),
        "name": item.get("name"),
        "normalized_name": item.get(
            "normalized_name"
        ),
        "item_id": item.get("item_id"),
        "internal_name": item.get(
            "internal_name"
        ),
    }


def _build_item_indexes(
    items: list[dict[str, Any]],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[int, list[dict[str, Any]]],
]:
    items_by_name: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    items_by_id: dict[
        int,
        list[dict[str, Any]],
    ] = defaultdict(list)

    seen_catalog_ids: set[str] = set()

    for item in items:
        source_catalog_id = item.get(
            "source_catalog_id"
        )

        if not source_catalog_id:
            raise ValueError(
                "Item record is missing "
                "source_catalog_id."
            )

        if source_catalog_id in seen_catalog_ids:
            raise ValueError(
                "Duplicate Item source_catalog_id: "
                f"{source_catalog_id}"
            )

        seen_catalog_ids.add(
            source_catalog_id
        )

        normalized_name = item.get(
            "normalized_name"
        )

        if normalized_name:
            items_by_name[
                str(normalized_name)
            ].append(item)

        item_id = item.get("item_id")

        if isinstance(item_id, int):
            items_by_id[
                item_id
            ].append(item)

    return (
        dict(items_by_name),
        dict(items_by_id),
    )


def _result_id_consistency(
    result: dict[str, Any],
    target_item: dict[str, Any],
) -> str:
    recipe_id_candidates = result.get(
        "item_id_candidates",
        [],
    )

    target_item_id = target_item.get(
        "item_id"
    )

    if not recipe_id_candidates:
        return "recipe_id_unavailable"

    if target_item_id is None:
        return "target_item_id_unavailable"

    if target_item_id in recipe_id_candidates:
        return "agree"

    return "conflict"


def _link_recipe_result(
    result: dict[str, Any],
    *,
    items_by_name: dict[
        str,
        list[dict[str, Any]],
    ],
    items_by_id: dict[
        int,
        list[dict[str, Any]],
    ],
) -> dict[str, Any]:
    normalized_name = result.get(
        "normalized_name"
    )

    recipe_id_candidates = [
        value
        for value in result.get(
            "item_id_candidates",
            [],
        )
        if isinstance(value, int)
    ]

    name_matches = (
        items_by_name.get(
            normalized_name,
            [],
        )
        if normalized_name
        else []
    )

    # ------------------------------------------------
    # Safe case 1:
    # one exact normalized-name match
    #
    # Even when an old recipe ID conflicts with a
    # modern ID, the exact name is still the safer
    # target than an ID-only match.
    # ------------------------------------------------

    if len(name_matches) == 1:
        target = name_matches[0]

        consistency = (
            _result_id_consistency(
                result,
                target,
            )
        )

        return {
            "status": "linked",
            "method": "exact_name",
            "id_consistency": consistency,
            "target": _item_reference(
                target
            ),
            "candidate_targets": [],
        }

    # ------------------------------------------------
    # Safe case 2:
    # several items have the same display name, but
    # the recipe ID selects exactly one of them.
    #
    # Example:
    # Constellation.
    # ------------------------------------------------

    if len(name_matches) > 1:
        id_selected_matches = [
            item
            for item in name_matches
            if item.get("item_id")
            in recipe_id_candidates
        ]

        if len(id_selected_matches) == 1:
            target = id_selected_matches[0]

            return {
                "status": "linked",
                "method": (
                    "exact_name_and_id_disambiguation"
                ),
                "id_consistency": "agree",
                "target": _item_reference(
                    target
                ),
                "candidate_targets": [
                    _item_reference(item)
                    for item in name_matches
                ],
            }

        return {
            "status": "ambiguous",
            "method": "exact_name_ambiguous",
            "id_consistency": "unresolved",
            "target": None,
            "candidate_targets": [
                _item_reference(item)
                for item in name_matches
            ],
        }

    # ------------------------------------------------
    # No name match:
    #
    # Do not auto-link by ID alone. Legacy recipe IDs
    # frequently collide with unrelated modern items.
    # ID matches are retained only as diagnostics.
    # ------------------------------------------------

    id_only_matches: list[
        dict[str, Any]
    ] = []

    seen_target_ids: set[str] = set()

    for candidate_id in recipe_id_candidates:
        for item in items_by_id.get(
            candidate_id,
            [],
        ):
            source_catalog_id = item.get(
                "source_catalog_id"
            )

            if source_catalog_id in seen_target_ids:
                continue

            seen_target_ids.add(
                source_catalog_id
            )

            id_only_matches.append(
                _item_reference(item)
            )

    return {
        "status": "unresolved",
        "method": "no_exact_name_match",
        "id_consistency": "not_applied",
        "target": None,
        "candidate_targets": (
            id_only_matches
        ),
    }


def _expand_alternative_members(
    group_name: str,
) -> list[str]:
    """
    Expand slash-separated alternatives while
    preserving a shared suffix.

    Examples:
        Adamantite/Titanium Bar
        ->
        Adamantite Bar
        Titanium Bar

        Adamantite Helmet/Titanium Helmet
        ->
        Adamantite Helmet
        Titanium Helmet
    """
    parts = [
        part.strip()
        for part in group_name.split("/")
        if part.strip()
    ]

    if len(parts) < 2:
        return parts

    reference_tokens = (
        parts[-1].split()
    )

    # The final alternative normally contains the
    # complete shared suffix, such as "Titanium Bar".
    if len(reference_tokens) <= 1:
        return parts

    shared_suffix = " ".join(
        reference_tokens[1:]
    )

    expanded_members: list[str] = []

    for part in parts:
        part_tokens = part.split()

        # A single-token alternative such as
        # "Adamantite" inherits "Bar" from
        # "Titanium Bar".
        if len(part_tokens) == 1:
            expanded = (
                f"{part} {shared_suffix}"
            ).strip()

        else:
            expanded = part

        if expanded not in expanded_members:
            expanded_members.append(
                expanded
            )

    return expanded_members


def _detect_ingredient_group(
    ingredient_name: str,
) -> dict[str, Any] | None:
    cleaned_name = ingredient_name.strip()
    folded_name = cleaned_name.casefold()

    # Any Wood, Any Iron Bar, Any Fruit, etc.
    #
    # This check must occur before slash handling so
    # "Any Seashell or Starfish" remains one semantic
    # recipe group.
    if folded_name.startswith("any "):
        group_label = cleaned_name[4:].strip()

        return {
            "group_type": "any_member",
            "name": cleaned_name,
            "member_description": (
                group_label
            ),
            "explicit_members": [],
        }

    # Adamantite/Titanium Bar,
    # Adamantite Helmet/Titanium Helmet, etc.
    if "/" in cleaned_name:
        members = (
            _expand_alternative_members(
                cleaned_name
            )
        )

        if len(members) >= 2:
            return {
                "group_type": "alternatives",
                "name": cleaned_name,
                "member_description": None,
                "explicit_members": members,
            }

    if folded_name == "recorded music boxes":
        return {
            "group_type": "collection",
            "name": cleaned_name,
            "member_description": (
                "Recorded Music Box"
            ),
            "explicit_members": [],
        }

    return None

def _link_ingredient(
    ingredient_item: dict[str, Any],
    *,
    items_by_name: dict[
        str,
        list[dict[str, Any]],
    ],
) -> dict[str, Any]:
    ingredient_name = str(
        ingredient_item.get(
            "name",
            "",
        )
    ).strip()

    normalized_name = ingredient_item.get(
        "normalized_name"
    )

    matches = (
        items_by_name.get(
            normalized_name,
            [],
        )
        if normalized_name
        else []
    )

    if len(matches) == 1:
        target = matches[0]

        return {
            "status": "linked",
            "kind": "item",
            "method": "exact_name",
            "target": _item_reference(
                target
            ),
            "group": None,
            "candidate_targets": [],
        }

    if len(matches) > 1:
        return {
            "status": "ambiguous",
            "kind": "item",
            "method": "exact_name_ambiguous",
            "target": None,
            "group": None,
            "candidate_targets": [
                _item_reference(item)
                for item in matches
            ],
        }

    group = _detect_ingredient_group(
        ingredient_name
    )

    if group is not None:
        return {
            "status": "group",
            "kind": "ingredient_group",
            "method": "group_syntax",
            "target": None,
            "group": group,
            "candidate_targets": [],
        }

    return {
        "status": "unresolved",
        "kind": "unresolved",
        "method": "no_exact_name_match",
        "target": None,
        "group": None,
        "candidate_targets": [],
    }


def _apply_ingredient_link(
    ingredient: dict[str, Any],
    link: dict[str, Any],
) -> None:
    ingredient_item = ingredient["item"]

    ingredient_item["kind"] = link[
        "kind"
    ]

    ingredient_item["link"] = link

    target = link.get("target")

    if target is not None:
        ingredient_item["item_id"] = (
            target["item_id"]
        )

        ingredient_item[
            "source_catalog_id"
        ] = target[
            "source_catalog_id"
        ]

        ingredient_item[
            "internal_name"
        ] = target[
            "internal_name"
        ]

    else:
        ingredient_item[
            "source_catalog_id"
        ] = None

        ingredient_item[
            "internal_name"
        ] = None


def link_recipe_record(
    recipe: dict[str, Any],
    *,
    items_by_name: dict[
        str,
        list[dict[str, Any]],
    ],
    items_by_id: dict[
        int,
        list[dict[str, Any]],
    ],
) -> dict[str, Any]:
    linked_recipe = copy.deepcopy(
        recipe
    )

    result_link = _link_recipe_result(
        linked_recipe["result"],
        items_by_name=items_by_name,
        items_by_id=items_by_id,
    )

    linked_recipe["result"][
        "item_catalog_link"
    ] = result_link

    ingredient_status_counts: Counter[
        str
    ] = Counter()

    unresolved_ingredient_names: set[
        str
    ] = set()

    ambiguous_ingredient_names: set[
        str
    ] = set()

    group_ingredient_names: set[
        str
    ] = set()

    for variant in linked_recipe[
        "variants"
    ]:
        for ingredient in variant[
            "ingredients"
        ]:
            ingredient_item = ingredient[
                "item"
            ]

            link = _link_ingredient(
                ingredient_item,
                items_by_name=items_by_name,
            )

            _apply_ingredient_link(
                ingredient,
                link,
            )

            status = link["status"]

            ingredient_status_counts[
                status
            ] += 1

            ingredient_name = (
                ingredient_item["name"]
            )

            if status == "unresolved":
                unresolved_ingredient_names.add(
                    ingredient_name
                )

            elif status == "ambiguous":
                ambiguous_ingredient_names.add(
                    ingredient_name
                )

            elif status == "group":
                group_ingredient_names.add(
                    ingredient_name
                )

    unresolved_count = (
        ingredient_status_counts[
            "unresolved"
        ]
        + ingredient_status_counts[
            "ambiguous"
        ]
    )

    result_is_resolved = (
        result_link["status"]
        == "linked"
    )

    if (
        result_is_resolved
        and unresolved_count == 0
    ):
        overall_status = "complete"

    else:
        overall_status = "partial"

    linked_recipe["linking"] = {
        "status": overall_status,

        "result_status": (
            result_link["status"]
        ),

        "ingredient_status_counts": dict(
            ingredient_status_counts
        ),

        "unresolved_ingredient_names": (
            sorted(
                unresolved_ingredient_names
            )
        ),

        "ambiguous_ingredient_names": (
            sorted(
                ambiguous_ingredient_names
            )
        ),

        "group_ingredient_names": (
            sorted(
                group_ingredient_names
            )
        ),
    }

    return linked_recipe


def link_recipes_file(
    items_path: str | Path = DEFAULT_ITEMS_PATH,
    recipes_path: str | Path = DEFAULT_RECIPES_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    items_path = Path(items_path)
    recipes_path = Path(recipes_path)
    output_path = Path(output_path)
    report_path = Path(report_path)

    items = _load_jsonl(
        items_path
    )

    recipes = _load_jsonl(
        recipes_path
    )

    (
        items_by_name,
        items_by_id,
    ) = _build_item_indexes(
        items
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
    complete_records = 0
    partial_records = 0
    total_variants = 0
    total_ingredient_entries = 0

    result_status_counts: Counter[str] = (
        Counter()
    )

    result_method_counts: Counter[str] = (
        Counter()
    )

    result_id_consistency_counts: Counter[
        str
    ] = Counter()

    ingredient_status_counts: Counter[
        str
    ] = Counter()

    ingredient_method_counts: Counter[
        str
    ] = Counter()

    ingredient_group_type_counts: Counter[
        str
    ] = Counter()

    unresolved_result_names: list[str] = []
    ambiguous_result_names: list[str] = []

    unresolved_ingredient_counts: Counter[
        str
    ] = Counter()

    ambiguous_ingredient_counts: Counter[
        str
    ] = Counter()

    ingredient_group_counts: Counter[
        str
    ] = Counter()

    seen_recipe_catalog_ids: set[str] = set()

    with temporary_output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        for recipe in recipes:
            source_catalog_id = recipe.get(
                "source_catalog_id"
            )

            if not source_catalog_id:
                raise ValueError(
                    "Recipe is missing "
                    "source_catalog_id."
                )

            if (
                source_catalog_id
                in seen_recipe_catalog_ids
            ):
                raise ValueError(
                    "Duplicate Recipe "
                    "source_catalog_id: "
                    f"{source_catalog_id}"
                )

            seen_recipe_catalog_ids.add(
                source_catalog_id
            )

            linked = link_recipe_record(
                recipe,
                items_by_name=items_by_name,
                items_by_id=items_by_id,
            )

            total_records += 1
            total_variants += len(
                linked["variants"]
            )

            if (
                linked["linking"]["status"]
                == "complete"
            ):
                complete_records += 1
            else:
                partial_records += 1

            result_link = linked[
                "result"
            ]["item_catalog_link"]

            result_status_counts.update(
                [
                    result_link["status"],
                ]
            )

            result_method_counts.update(
                [
                    result_link["method"],
                ]
            )

            result_id_consistency_counts.update(
                [
                    result_link[
                        "id_consistency"
                    ],
                ]
            )

            result_name = linked[
                "result"
            ]["name"]

            if (
                result_link["status"]
                == "unresolved"
            ):
                unresolved_result_names.append(
                    result_name
                )

            elif (
                result_link["status"]
                == "ambiguous"
            ):
                ambiguous_result_names.append(
                    result_name
                )

            for variant in linked["variants"]:
                for ingredient in variant[
                    "ingredients"
                ]:
                    total_ingredient_entries += 1

                    ingredient_item = (
                        ingredient["item"]
                    )

                    ingredient_link = (
                        ingredient_item["link"]
                    )

                    ingredient_status_counts.update(
                        [
                            ingredient_link[
                                "status"
                            ],
                        ]
                    )

                    ingredient_method_counts.update(
                        [
                            ingredient_link[
                                "method"
                            ],
                        ]
                    )

                    ingredient_name = (
                        ingredient_item["name"]
                    )

                    if (
                        ingredient_link["status"]
                        == "unresolved"
                    ):
                        unresolved_ingredient_counts[
                            ingredient_name
                        ] += 1

                    elif (
                        ingredient_link["status"]
                        == "ambiguous"
                    ):
                        ambiguous_ingredient_counts[
                            ingredient_name
                        ] += 1

                    elif (
                        ingredient_link["status"]
                        == "group"
                    ):
                        ingredient_group_counts[
                            ingredient_name
                        ] += 1

                        group_type = (
                            ingredient_link[
                                "group"
                            ]["group_type"]
                        )

                        ingredient_group_type_counts.update(
                            [
                                group_type,
                            ]
                        )

            output_file.write(
                json.dumps(
                    linked,
                    ensure_ascii=False,
                )
                + "\n"
            )

    if total_records == 0:
        temporary_output_path.unlink(
            missing_ok=True
        )

        raise ValueError(
            "No Recipe records were linked."
        )

    temporary_output_path.replace(
        output_path
    )

    directly_linked_ingredients = (
        ingredient_status_counts[
            "linked"
        ]
    )

    usable_ingredient_entries = (
        ingredient_status_counts[
            "linked"
        ]
        + ingredient_status_counts[
            "group"
        ]
    )

    report = {
        "items_path": str(items_path),
        "recipes_path": str(
            recipes_path
        ),
        "output_path": str(output_path),

        "item_records": len(items),
        "recipe_records": total_records,
        "complete_records": (
            complete_records
        ),
        "partial_records": partial_records,

        "total_variants": total_variants,
        "total_ingredient_entries": (
            total_ingredient_entries
        ),

        "result_status_counts": dict(
            result_status_counts.most_common()
        ),

        "result_method_counts": dict(
            result_method_counts.most_common()
        ),

        "result_id_consistency_counts": dict(
            result_id_consistency_counts.most_common()
        ),

        "ingredient_status_counts": dict(
            ingredient_status_counts.most_common()
        ),

        "ingredient_method_counts": dict(
            ingredient_method_counts.most_common()
        ),

        "ingredient_group_type_counts": dict(
            ingredient_group_type_counts.most_common()
        ),

        "direct_ingredient_link_coverage_percent": (
            round(
                100.0
                * directly_linked_ingredients
                / total_ingredient_entries,
                4,
            )
        ),

        "usable_ingredient_coverage_percent": (
            round(
                100.0
                * usable_ingredient_entries
                / total_ingredient_entries,
                4,
            )
        ),

        "unresolved_result_names": sorted(
            unresolved_result_names
        ),

        "ambiguous_result_names": sorted(
            ambiguous_result_names
        ),

        "unresolved_ingredient_counts": dict(
            unresolved_ingredient_counts.most_common()
        ),

        "ambiguous_ingredient_counts": dict(
            ambiguous_ingredient_counts.most_common()
        ),

        "ingredient_group_counts": dict(
            ingredient_group_counts.most_common()
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
