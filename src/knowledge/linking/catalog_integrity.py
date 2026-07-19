
from __future__ import annotations

import hashlib
import json

from collections import Counter
from pathlib import Path
from typing import Any


CATALOG_ROOT = Path(
    "/content/llm_project/data/terraria/catalog"
)

DEFAULT_ITEMS_PATH = (
    CATALOG_ROOT
    / "cleaned"
    / "Items.jsonl"
)

DEFAULT_NPCS_PATH = (
    CATALOG_ROOT
    / "cleaned"
    / "NPCs.jsonl"
)

DEFAULT_CLEANED_RECIPES_PATH = (
    CATALOG_ROOT
    / "cleaned"
    / "Recipes.jsonl"
)

DEFAULT_LINKED_RECIPES_PATH = (
    CATALOG_ROOT
    / "linked"
    / "Recipes.jsonl"
)

DEFAULT_CLEANED_DROPS_PATH = (
    CATALOG_ROOT
    / "cleaned"
    / "Drops.jsonl"
)

DEFAULT_LINKED_DROPS_PATH = (
    CATALOG_ROOT
    / "linked"
    / "Drops.jsonl"
)

DEFAULT_REPORT_PATH = (
    CATALOG_ROOT
    / "linked"
    / "catalog_integrity_report.json"
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
                    f"Record at "
                    f"{path}:{line_number} "
                    "is not a dictionary."
                )

            records.append(record)

    return records


def _sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def _index_records(
    records: list[dict[str, Any]],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    index: dict[
        str,
        dict[str, Any],
    ] = {}

    for record in records:
        source_catalog_id = record.get(
            "source_catalog_id"
        )

        if not source_catalog_id:
            raise ValueError(
                f"{label} record is missing "
                "source_catalog_id."
            )

        source_catalog_id = str(
            source_catalog_id
        )

        if source_catalog_id in index:
            raise ValueError(
                f"Duplicate {label} "
                "source_catalog_id: "
                f"{source_catalog_id}"
            )

        index[source_catalog_id] = record

    return index


def _validate_item_reference(
    reference: dict[str, Any],
    item_index: dict[
        str,
        dict[str, Any],
    ],
    *,
    context: str,
) -> None:
    source_catalog_id = reference.get(
        "source_catalog_id"
    )

    if source_catalog_id not in item_index:
        raise AssertionError(
            f"Dangling Item reference in "
            f"{context}: "
            f"{source_catalog_id}"
        )

    canonical = item_index[
        source_catalog_id
    ]

    for field in (
        "name",
        "normalized_name",
        "item_id",
        "internal_name",
    ):
        if reference.get(field) != canonical.get(
            field
        ):
            raise AssertionError(
                f"Item reference mismatch in "
                f"{context}, field={field}: "
                f"reference={reference.get(field)!r}, "
                f"canonical={canonical.get(field)!r}"
            )


def _validate_npc_reference(
    reference: dict[str, Any],
    npc_index: dict[
        str,
        dict[str, Any],
    ],
    *,
    context: str,
) -> None:
    source_catalog_id = reference.get(
        "source_catalog_id"
    )

    if source_catalog_id not in npc_index:
        raise AssertionError(
            f"Dangling NPC reference in "
            f"{context}: "
            f"{source_catalog_id}"
        )

    canonical = npc_index[
        source_catalog_id
    ]

    for field in (
        "name",
        "normalized_name",
        "npc_id",
    ):
        if reference.get(field) != canonical.get(
            field
        ):
            raise AssertionError(
                f"NPC reference mismatch in "
                f"{context}, field={field}: "
                f"reference={reference.get(field)!r}, "
                f"canonical={canonical.get(field)!r}"
            )


def _validate_item_candidates(
    candidates: list[dict[str, Any]],
    item_index: dict[
        str,
        dict[str, Any],
    ],
    *,
    context: str,
) -> int:
    count = 0

    for position, candidate in enumerate(
        candidates
    ):
        _validate_item_reference(
            candidate,
            item_index,
            context=(
                f"{context}.candidate_targets"
                f"[{position}]"
            ),
        )

        count += 1

    return count


def _validate_npc_candidates(
    candidates: list[dict[str, Any]],
    npc_index: dict[
        str,
        dict[str, Any],
    ],
    *,
    context: str,
) -> int:
    count = 0

    for position, candidate in enumerate(
        candidates
    ):
        _validate_npc_reference(
            candidate,
            npc_index,
            context=(
                f"{context}.candidate_targets"
                f"[{position}]"
            ),
        )

        count += 1

    return count


def audit_catalog_integrity(
    items_path: str | Path = (
        DEFAULT_ITEMS_PATH
    ),
    npcs_path: str | Path = (
        DEFAULT_NPCS_PATH
    ),
    cleaned_recipes_path: str | Path = (
        DEFAULT_CLEANED_RECIPES_PATH
    ),
    linked_recipes_path: str | Path = (
        DEFAULT_LINKED_RECIPES_PATH
    ),
    cleaned_drops_path: str | Path = (
        DEFAULT_CLEANED_DROPS_PATH
    ),
    linked_drops_path: str | Path = (
        DEFAULT_LINKED_DROPS_PATH
    ),
    report_path: str | Path = (
        DEFAULT_REPORT_PATH
    ),
) -> dict[str, Any]:
    items_path = Path(items_path)
    npcs_path = Path(npcs_path)

    cleaned_recipes_path = Path(
        cleaned_recipes_path
    )

    linked_recipes_path = Path(
        linked_recipes_path
    )

    cleaned_drops_path = Path(
        cleaned_drops_path
    )

    linked_drops_path = Path(
        linked_drops_path
    )

    report_path = Path(report_path)

    items = _load_jsonl(
        items_path
    )

    npcs = _load_jsonl(
        npcs_path
    )

    cleaned_recipes = _load_jsonl(
        cleaned_recipes_path
    )

    linked_recipes = _load_jsonl(
        linked_recipes_path
    )

    cleaned_drops = _load_jsonl(
        cleaned_drops_path
    )

    linked_drops = _load_jsonl(
        linked_drops_path
    )

    item_index = _index_records(
        items,
        label="Item",
    )

    npc_index = _index_records(
        npcs,
        label="NPC",
    )

    cleaned_recipe_index = _index_records(
        cleaned_recipes,
        label="cleaned Recipe",
    )

    linked_recipe_index = _index_records(
        linked_recipes,
        label="linked Recipe",
    )

    cleaned_drop_index = _index_records(
        cleaned_drops,
        label="cleaned Drop",
    )

    linked_drop_index = _index_records(
        linked_drops,
        label="linked Drop",
    )

    # ----------------------------------------------
    # Linked layers must preserve the exact record
    # identity set of their cleaned source layers.
    # ----------------------------------------------

    if (
        set(cleaned_recipe_index)
        != set(linked_recipe_index)
    ):
        raise AssertionError(
            "Cleaned and linked Recipe identity "
            "sets differ."
        )

    if (
        set(cleaned_drop_index)
        != set(linked_drop_index)
    ):
        raise AssertionError(
            "Cleaned and linked Drop identity "
            "sets differ."
        )

    recipe_result_status_counts: Counter[
        str
    ] = Counter()

    recipe_ingredient_status_counts: Counter[
        str
    ] = Counter()

    recipe_linking_status_counts: Counter[
        str
    ] = Counter()

    drop_item_status_counts: Counter[
        str
    ] = Counter()

    drop_source_status_counts: Counter[
        str
    ] = Counter()

    drop_linking_status_counts: Counter[
        str
    ] = Counter()

    recipe_parse_status_counts: Counter[
        str
    ] = Counter()

    drop_parse_status_counts: Counter[
        str
    ] = Counter()

    resolved_item_references = 0
    resolved_npc_references = 0

    validated_item_candidates = 0
    validated_npc_candidates = 0

    recipe_variant_count = 0
    recipe_ingredient_count = 0

    referenced_item_catalog_ids: set[
        str
    ] = set()

    referenced_npc_catalog_ids: set[
        str
    ] = set()

    # ----------------------------------------------
    # Audit linked Recipes.
    # ----------------------------------------------

    for recipe in linked_recipes:
        recipe_id = recipe[
            "source_catalog_id"
        ]

        recipe_parse_status_counts[
            recipe["parse_status"]
        ] += 1

        linking_status = recipe[
            "linking"
        ]["status"]

        recipe_linking_status_counts[
            linking_status
        ] += 1

        result_link = recipe[
            "result"
        ]["item_catalog_link"]

        result_status = result_link[
            "status"
        ]

        recipe_result_status_counts[
            result_status
        ] += 1

        result_target = result_link.get(
            "target"
        )

        if result_status == "linked":
            if not isinstance(
                result_target,
                dict,
            ):
                raise AssertionError(
                    f"Linked Recipe result has no "
                    f"target: {recipe_id}"
                )

            _validate_item_reference(
                result_target,
                item_index,
                context=(
                    f"Recipe {recipe_id} result"
                ),
            )

            resolved_item_references += 1

            referenced_item_catalog_ids.add(
                result_target[
                    "source_catalog_id"
                ]
            )

        elif result_target is not None:
            raise AssertionError(
                f"Non-linked Recipe result has "
                f"a target: {recipe_id}"
            )

        validated_item_candidates += (
            _validate_item_candidates(
                result_link.get(
                    "candidate_targets",
                    [],
                ),
                item_index,
                context=(
                    f"Recipe {recipe_id} result"
                ),
            )
        )

        for variant in recipe["variants"]:
            recipe_variant_count += 1

            for ingredient in variant[
                "ingredients"
            ]:
                recipe_ingredient_count += 1

                ingredient_item = ingredient[
                    "item"
                ]

                ingredient_link = (
                    ingredient_item["link"]
                )

                ingredient_status = (
                    ingredient_link["status"]
                )

                recipe_ingredient_status_counts[
                    ingredient_status
                ] += 1

                target = ingredient_link.get(
                    "target"
                )

                if ingredient_status == "linked":
                    if not isinstance(
                        target,
                        dict,
                    ):
                        raise AssertionError(
                            "Linked Recipe ingredient "
                            "has no target: "
                            f"{recipe_id}/"
                            f"{variant['variant_id']}"
                        )

                    _validate_item_reference(
                        target,
                        item_index,
                        context=(
                            f"Recipe {recipe_id} "
                            f"variant "
                            f"{variant['variant_id']} "
                            "ingredient "
                            f"{ingredient_item['name']}"
                        ),
                    )

                    if (
                        ingredient_item.get(
                            "item_id"
                        )
                        != target.get("item_id")
                    ):
                        raise AssertionError(
                            "Recipe ingredient item_id "
                            "does not match target."
                        )

                    if (
                        ingredient_item.get(
                            "source_catalog_id"
                        )
                        != target.get(
                            "source_catalog_id"
                        )
                    ):
                        raise AssertionError(
                            "Recipe ingredient "
                            "source_catalog_id does not "
                            "match target."
                        )

                    resolved_item_references += 1

                    referenced_item_catalog_ids.add(
                        target[
                            "source_catalog_id"
                        ]
                    )

                elif target is not None:
                    raise AssertionError(
                        "Non-linked Recipe ingredient "
                        "has a target."
                    )

                validated_item_candidates += (
                    _validate_item_candidates(
                        ingredient_link.get(
                            "candidate_targets",
                            [],
                        ),
                        item_index,
                        context=(
                            f"Recipe {recipe_id} "
                            f"variant "
                            f"{variant['variant_id']} "
                            "ingredient candidates"
                        ),
                    )
                )

    # ----------------------------------------------
    # Audit linked Drops.
    # ----------------------------------------------

    for drop in linked_drops:
        drop_id = drop[
            "source_catalog_id"
        ]

        drop_parse_status_counts[
            drop["parse_status"]
        ] += 1

        drop_linking_status = drop[
            "linking"
        ]["status"]

        drop_linking_status_counts[
            drop_linking_status
        ] += 1

        item_data = drop["item"]
        item_link = item_data[
            "catalog_link"
        ]

        item_status = item_link["status"]

        drop_item_status_counts[
            item_status
        ] += 1

        item_target = item_link.get(
            "target"
        )

        if item_status == "linked":
            if not isinstance(
                item_target,
                dict,
            ):
                raise AssertionError(
                    f"Linked Drop item has no "
                    f"target: {drop_id}"
                )

            _validate_item_reference(
                item_target,
                item_index,
                context=(
                    f"Drop {drop_id} item"
                ),
            )

            if (
                item_data.get("item_id")
                != item_target.get("item_id")
            ):
                raise AssertionError(
                    f"Drop item_id mismatch: "
                    f"{drop_id}"
                )

            if (
                item_data.get(
                    "source_catalog_id"
                )
                != item_target.get(
                    "source_catalog_id"
                )
            ):
                raise AssertionError(
                    "Drop Item source_catalog_id "
                    f"mismatch: {drop_id}"
                )

            resolved_item_references += 1

            referenced_item_catalog_ids.add(
                item_target[
                    "source_catalog_id"
                ]
            )

        elif item_target is not None:
            raise AssertionError(
                f"Non-linked Drop item has a "
                f"target: {drop_id}"
            )

        validated_item_candidates += (
            _validate_item_candidates(
                item_link.get(
                    "candidate_targets",
                    [],
                ),
                item_index,
                context=(
                    f"Drop {drop_id} item"
                ),
            )
        )

        source_data = drop["source"]
        source_link = source_data[
            "catalog_link"
        ]

        source_status = source_link[
            "status"
        ]

        drop_source_status_counts[
            source_status
        ] += 1

        source_target = source_link.get(
            "target"
        )

        if source_status == "linked":
            if not isinstance(
                source_target,
                dict,
            ):
                raise AssertionError(
                    f"Linked Drop source has no "
                    f"target: {drop_id}"
                )

            _validate_npc_reference(
                source_target,
                npc_index,
                context=(
                    f"Drop {drop_id} source"
                ),
            )

            if (
                source_data.get(
                    "resolved_npc_id"
                )
                != source_target.get(
                    "npc_id"
                )
            ):
                raise AssertionError(
                    f"Drop resolved_npc_id "
                    f"mismatch: {drop_id}"
                )

            if (
                source_data.get(
                    "npc_source_catalog_id"
                )
                != source_target.get(
                    "source_catalog_id"
                )
            ):
                raise AssertionError(
                    "Drop NPC source_catalog_id "
                    f"mismatch: {drop_id}"
                )

            resolved_npc_references += 1

            referenced_npc_catalog_ids.add(
                source_target[
                    "source_catalog_id"
                ]
            )

        elif source_target is not None:
            raise AssertionError(
                f"Non-linked Drop source has "
                f"a target: {drop_id}"
            )

        validated_npc_candidates += (
            _validate_npc_candidates(
                source_link.get(
                    "candidate_targets",
                    [],
                ),
                npc_index,
                context=(
                    f"Drop {drop_id} source"
                ),
            )
        )

        # NPC-family groups contain concrete NPC
        # references inside explicit_members.
        source_group = source_link.get(
            "group"
        )

        if (
            source_link.get("kind")
            == "npc_family"
        ):
            if not isinstance(
                source_group,
                dict,
            ):
                raise AssertionError(
                    f"NPC family has no group: "
                    f"{drop_id}"
                )

            members = source_group.get(
                "explicit_members",
                [],
            )

            if not members:
                raise AssertionError(
                    f"NPC family has no members: "
                    f"{drop_id}"
                )

            for position, member in enumerate(
                members
            ):
                _validate_npc_reference(
                    member,
                    npc_index,
                    context=(
                        f"Drop {drop_id} "
                        f"npc_family member "
                        f"{position}"
                    ),
                )

                validated_npc_candidates += 1

    # ----------------------------------------------
    # Exact expected state after the cleaning and
    # linking pipeline.
    # ----------------------------------------------

    expected_recipe_result_statuses = {
        "linked": 3317,
        "unresolved": 92,
    }

    expected_recipe_ingredient_statuses = {
        "linked": 6537,
        "group": 377,
        "unresolved": 45,
    }

    expected_recipe_linking_statuses = {
        "complete": 3304,
        "partial": 105,
    }

    expected_drop_item_statuses = {
        "linked": 3131,
        "unresolved": 8,
        "group": 5,
    }

    expected_drop_source_statuses = {
        "classified": 1694,
        "linked": 1368,
        "group": 80,
        "unresolved": 2,
    }

    expected_drop_linking_statuses = {
        "complete": 3134,
        "partial": 10,
    }

    if dict(
        recipe_result_status_counts
    ) != expected_recipe_result_statuses:
        raise AssertionError(
            "Unexpected Recipe result "
            "link-status counts."
        )

    if dict(
        recipe_ingredient_status_counts
    ) != expected_recipe_ingredient_statuses:
        raise AssertionError(
            "Unexpected Recipe ingredient "
            "link-status counts."
        )

    if dict(
        recipe_linking_status_counts
    ) != expected_recipe_linking_statuses:
        raise AssertionError(
            "Unexpected Recipe overall "
            "link-status counts."
        )

    if dict(
        drop_item_status_counts
    ) != expected_drop_item_statuses:
        raise AssertionError(
            "Unexpected Drop item "
            "link-status counts."
        )

    if dict(
        drop_source_status_counts
    ) != expected_drop_source_statuses:
        raise AssertionError(
            "Unexpected Drop source "
            "link-status counts."
        )

    if dict(
        drop_linking_status_counts
    ) != expected_drop_linking_statuses:
        raise AssertionError(
            "Unexpected Drop overall "
            "link-status counts."
        )

    if recipe_variant_count != 4221:
        raise AssertionError(
            "Unexpected Recipe variant count."
        )

    if recipe_ingredient_count != 6959:
        raise AssertionError(
            "Unexpected Recipe ingredient count."
        )

    # Resolved foreign-key references:
    #
    # Recipe result Items:      3317
    # Recipe ingredient Items:  6537
    # Drop Items:               3131
    # Drop NPC sources:         1368
    expected_resolved_references = (
        3317
        + 6537
        + 3131
        + 1368
    )

    actual_resolved_references = (
        resolved_item_references
        + resolved_npc_references
    )

    if (
        actual_resolved_references
        != expected_resolved_references
    ):
        raise AssertionError(
            "Unexpected total resolved "
            "reference count."
        )

    report = {
        "status": "passed",

        "record_counts": {
            "items": len(items),
            "npcs": len(npcs),
            "cleaned_recipes": len(
                cleaned_recipes
            ),
            "linked_recipes": len(
                linked_recipes
            ),
            "cleaned_drops": len(
                cleaned_drops
            ),
            "linked_drops": len(
                linked_drops
            ),
        },

        "identity_parity": {
            "recipes": True,
            "drops": True,
        },

        "recipe_counts": {
            "variants": (
                recipe_variant_count
            ),
            "ingredients": (
                recipe_ingredient_count
            ),
            "result_statuses": dict(
                recipe_result_status_counts
            ),
            "ingredient_statuses": dict(
                recipe_ingredient_status_counts
            ),
            "linking_statuses": dict(
                recipe_linking_status_counts
            ),
            "parse_statuses": dict(
                recipe_parse_status_counts
            ),
        },

        "drop_counts": {
            "item_statuses": dict(
                drop_item_status_counts
            ),
            "source_statuses": dict(
                drop_source_status_counts
            ),
            "linking_statuses": dict(
                drop_linking_status_counts
            ),
            "parse_statuses": dict(
                drop_parse_status_counts
            ),
        },

        "reference_integrity": {
            "resolved_item_references": (
                resolved_item_references
            ),
            "resolved_npc_references": (
                resolved_npc_references
            ),
            "resolved_references_total": (
                actual_resolved_references
            ),
            "validated_item_candidates": (
                validated_item_candidates
            ),
            "validated_npc_candidates": (
                validated_npc_candidates
            ),
            "dangling_item_references": 0,
            "dangling_npc_references": 0,
            "mismatched_item_references": 0,
            "mismatched_npc_references": 0,
        },

        "referenced_entities": {
            "unique_items": len(
                referenced_item_catalog_ids
            ),
            "unique_npcs": len(
                referenced_npc_catalog_ids
            ),
        },

        "file_sha256": {
            "Items.jsonl": _sha256(
                items_path
            ),
            "NPCs.jsonl": _sha256(
                npcs_path
            ),
            "cleaned/Recipes.jsonl": (
                _sha256(
                    cleaned_recipes_path
                )
            ),
            "linked/Recipes.jsonl": (
                _sha256(
                    linked_recipes_path
                )
            ),
            "cleaned/Drops.jsonl": (
                _sha256(
                    cleaned_drops_path
                )
            ),
            "linked/Drops.jsonl": (
                _sha256(
                    linked_drops_path
                )
            ),
        },
    }

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
