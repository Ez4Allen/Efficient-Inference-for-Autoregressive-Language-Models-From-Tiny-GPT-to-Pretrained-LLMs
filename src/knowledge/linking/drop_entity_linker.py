
from __future__ import annotations

import copy
import json
import re

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..catalog_store import normalize_catalog_name


DEFAULT_ITEMS_PATH = Path(
    "/content/llm_project/data/terraria/"
    "catalog/cleaned/Items.jsonl"
)

DEFAULT_NPCS_PATH = Path(
    "/content/llm_project/data/terraria/"
    "catalog/cleaned/NPCs.jsonl"
)

DEFAULT_DROPS_PATH = Path(
    "/content/llm_project/data/terraria/"
    "catalog/cleaned/Drops.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "/content/llm_project/data/terraria/"
    "catalog/linked/Drops.jsonl"
)

DEFAULT_REPORT_PATH = Path(
    "/content/llm_project/data/terraria/"
    "catalog/linked/Drops_link_report.json"
)


ITEM_QUALIFIER_PATTERN = re.compile(
    r"\s+\((item|painting)\)\s*$",
    flags=re.IGNORECASE,
)

BONUS_DROP_PATTERN = re.compile(
    r"\s+\(bonus drop\)\s*$",
    flags=re.IGNORECASE,
)


# Exact duplicate Item names that can be resolved
# deterministically from their Terraria semantics.
EXACT_ITEM_INTERNAL_NAME_RULES = {
    "Seaweed": "Seaweed",
    "Ogre Mask": "BossMaskOgre",
}


WORLD_OBJECT_SOURCES = {
    "Shadow Orb",
    "Crimson Heart",
}


COMPOSITE_NPC_GROUPS = {
    "The Twins": [
        "Retinazer",
        "Spazmatism",
    ],

    "Mechdusa": [
        "The Destroyer",
        "Skeletron Prime",
        "Retinazer",
        "Spazmatism",
    ],
}


ITEM_GROUPS = {
    "Golden furniture": {
        "group_type": "item_collection",
        "member_description": (
            "Golden furniture item"
        ),
        "explicit_members": [],
    },
}


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


def _build_name_index(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    index: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for record in records:
        normalized_name = record.get(
            "normalized_name"
        )

        if normalized_name:
            index[
                str(normalized_name)
            ].append(record)

    return dict(index)


def _build_npc_id_index(
    npcs: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    index: dict[
        int,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for npc in npcs:
        npc_id = npc.get("npc_id")

        if isinstance(npc_id, int):
            index[npc_id].append(npc)

    return dict(index)


def _item_reference(
    item: dict[str, Any],
) -> dict[str, Any]:
    classification = item.get(
        "classification"
    )

    if not isinstance(
        classification,
        dict,
    ):
        classification = {}

    types = classification.get(
        "types",
        [],
    )

    if not isinstance(types, list):
        types = []

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
        "types": types,
    }


def _npc_reference(
    npc: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source_catalog_id": npc.get(
            "source_catalog_id"
        ),
        "name": npc.get("name"),
        "normalized_name": npc.get(
            "normalized_name"
        ),
        "npc_id": npc.get("npc_id"),
        "npc_types": npc.get(
            "npc_types",
            [],
        ),
        "environment": npc.get(
            "environment",
            [],
        ),
    }


def _parse_qualified_item_name(
    item_name: str,
) -> tuple[str, str | None]:
    match = ITEM_QUALIFIER_PATTERN.search(
        item_name
    )

    if not match:
        return item_name, None

    base_name = item_name[
        :match.start()
    ].strip()

    qualifier = match.group(
        1
    ).casefold()

    return base_name, qualifier


def _select_item_by_qualifier(
    matches: list[dict[str, Any]],
    qualifier: str,
) -> list[dict[str, Any]]:
    if qualifier == "painting":
        selected = []

        for item in matches:
            classification = item.get(
                "classification",
                {},
            )

            if not isinstance(
                classification,
                dict,
            ):
                continue

            types = classification.get(
                "types",
                [],
            )

            if not isinstance(types, list):
                continue

            folded_types = {
                str(item_type).casefold()
                for item_type in types
            }

            if "furniture" in folded_types:
                selected.append(item)

        return selected

    return matches


def _select_item_by_internal_name_rule(
    item_name: str,
    matches: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expected_internal_name = (
        EXACT_ITEM_INTERNAL_NAME_RULES.get(
            item_name
        )
    )

    if not expected_internal_name:
        return []

    return [
        item
        for item in matches
        if item.get("internal_name")
        == expected_internal_name
    ]


def _link_drop_item(
    item_data: dict[str, Any],
    *,
    items_by_name: dict[
        str,
        list[dict[str, Any]],
    ],
) -> dict[str, Any]:
    item_name = str(
        item_data.get(
            "name",
            "",
        )
    ).strip()

    normalized_name = item_data.get(
        "normalized_name"
    )

    # ------------------------------------------------
    # Semantic item collection.
    # ------------------------------------------------

    if item_name in ITEM_GROUPS:
        group_definition = copy.deepcopy(
            ITEM_GROUPS[item_name]
        )

        group_definition["name"] = (
            item_name
        )

        return {
            "status": "group",
            "kind": "item_group",
            "method": "known_item_group",
            "target": None,
            "group": group_definition,
            "candidate_targets": [],
        }

    # ------------------------------------------------
    # Exact normalized-name lookup.
    # ------------------------------------------------

    exact_matches = (
        items_by_name.get(
            str(normalized_name),
            [],
        )
        if normalized_name
        else []
    )

    if len(exact_matches) == 1:
        return {
            "status": "linked",
            "kind": "item",
            "method": "exact_name",
            "target": _item_reference(
                exact_matches[0]
            ),
            "group": None,
            "candidate_targets": [],
        }

    if len(exact_matches) > 1:
        selected = (
            _select_item_by_internal_name_rule(
                item_name,
                exact_matches,
            )
        )

        if len(selected) == 1:
            return {
                "status": "linked",
                "kind": "item",
                "method": (
                    "exact_name_catalog_rule"
                ),
                "target": _item_reference(
                    selected[0]
                ),
                "group": None,
                "candidate_targets": [
                    _item_reference(item)
                    for item in exact_matches
                ],
            }

        return {
            "status": "ambiguous",
            "kind": "item",
            "method": "exact_name_ambiguous",
            "target": None,
            "group": None,
            "candidate_targets": [
                _item_reference(item)
                for item in exact_matches
            ],
        }

    # ------------------------------------------------
    # Cargo qualifiers:
    #
    # Shadow Orb (item)
    # Constellation (painting)
    # ------------------------------------------------

    (
        base_name,
        qualifier,
    ) = _parse_qualified_item_name(
        item_name
    )

    if qualifier:
        base_matches = items_by_name.get(
            normalize_catalog_name(
                base_name
            ),
            [],
        )

        if len(base_matches) == 1:
            return {
                "status": "linked",
                "kind": "item",
                "method": (
                    "qualified_alias"
                ),
                "target": _item_reference(
                    base_matches[0]
                ),
                "group": None,
                "candidate_targets": [],
                "alias": {
                    "original_name": item_name,
                    "base_name": base_name,
                    "qualifier": qualifier,
                },
            }

        if len(base_matches) > 1:
            selected = (
                _select_item_by_qualifier(
                    base_matches,
                    qualifier,
                )
            )

            if len(selected) == 1:
                return {
                    "status": "linked",
                    "kind": "item",
                    "method": (
                        "qualified_alias_disambiguation"
                    ),
                    "target": _item_reference(
                        selected[0]
                    ),
                    "group": None,
                    "candidate_targets": [
                        _item_reference(item)
                        for item in base_matches
                    ],
                    "alias": {
                        "original_name": item_name,
                        "base_name": base_name,
                        "qualifier": qualifier,
                    },
                }

            return {
                "status": "ambiguous",
                "kind": "item",
                "method": (
                    "qualified_alias_ambiguous"
                ),
                "target": None,
                "group": None,
                "candidate_targets": [
                    _item_reference(item)
                    for item in base_matches
                ],
                "alias": {
                    "original_name": item_name,
                    "base_name": base_name,
                    "qualifier": qualifier,
                },
            }

    # ------------------------------------------------
    # Legacy or catalog-missing item.
    # ------------------------------------------------

    return {
        "status": "unresolved",
        "kind": "legacy_or_missing_item",
        "method": "no_safe_item_match",
        "target": None,
        "group": None,
        "candidate_targets": [],
    }


def _apply_item_link(
    item_data: dict[str, Any],
    link: dict[str, Any],
) -> None:
    item_data["kind"] = link["kind"]
    item_data["catalog_link"] = link

    target = link.get("target")

    if target is None:
        item_data["item_id"] = None
        item_data[
            "source_catalog_id"
        ] = None

        item_data[
            "internal_name"
        ] = None

        return

    item_data["item_id"] = target[
        "item_id"
    ]

    item_data[
        "source_catalog_id"
    ] = target[
        "source_catalog_id"
    ]

    item_data[
        "internal_name"
    ] = target[
        "internal_name"
    ]


def _link_bonus_drop_alias(
    source_name: str,
    source_id: int | None,
    *,
    npcs_by_name: dict[
        str,
        list[dict[str, Any]],
    ],
) -> dict[str, Any] | None:
    match = BONUS_DROP_PATTERN.search(
        source_name
    )

    if not match:
        return None

    base_name = source_name[
        :match.start()
    ].strip()

    base_matches = npcs_by_name.get(
        normalize_catalog_name(
            base_name
        ),
        [],
    )

    id_selected = [
        npc
        for npc in base_matches
        if npc.get("npc_id")
        == source_id
    ]

    if len(id_selected) == 1:
        target = id_selected[0]

    elif len(base_matches) == 1:
        target = base_matches[0]

    else:
        return {
            "status": "ambiguous",
            "kind": "npc",
            "method": (
                "bonus_drop_alias_ambiguous"
            ),
            "target": None,
            "group": None,
            "candidate_targets": [
                _npc_reference(npc)
                for npc in base_matches
            ],
            "alias": {
                "original_name": source_name,
                "base_name": base_name,
                "qualifier": "bonus drop",
            },
        }

    return {
        "status": "linked",
        "kind": "npc",
        "method": "bonus_drop_alias",
        "target": _npc_reference(
            target
        ),
        "group": None,
        "candidate_targets": [],
        "alias": {
            "original_name": source_name,
            "base_name": base_name,
            "qualifier": "bonus drop",
        },
    }


def _link_npc_source(
    source_data: dict[str, Any],
    *,
    npcs_by_name: dict[
        str,
        list[dict[str, Any]],
    ],
    npcs_by_id: dict[
        int,
        list[dict[str, Any]],
    ],
) -> dict[str, Any]:
    source_name = str(
        source_data.get(
            "name",
            "",
        )
    ).strip()

    normalized_name = source_data.get(
        "normalized_name"
    )

    source_id = source_data.get(
        "source_id"
    )

    bonus_link = _link_bonus_drop_alias(
        source_name,
        source_id,
        npcs_by_name=npcs_by_name,
    )

    if bonus_link is not None:
        return bonus_link

    # ------------------------------------------------
    # World object represented by Cargo as NPC source.
    # ------------------------------------------------

    if source_name in WORLD_OBJECT_SOURCES:
        return {
            "status": "group",
            "kind": "world_object",
            "method": "known_world_object",
            "target": None,
            "group": {
                "group_type": "world_object",
                "name": source_name,
                "explicit_members": [],
            },
            "candidate_targets": [],
        }

    # ------------------------------------------------
    # Composite boss source.
    # ------------------------------------------------

    if source_name in COMPOSITE_NPC_GROUPS:
        return {
            "status": "group",
            "kind": "composite_npc_group",
            "method": (
                "known_composite_npc_group"
            ),
            "target": None,
            "group": {
                "group_type": (
                    "composite_npc_group"
                ),
                "name": source_name,
                "explicit_members": (
                    COMPOSITE_NPC_GROUPS[
                        source_name
                    ]
                ),
            },
            "candidate_targets": [],
        }

    name_matches = (
        npcs_by_name.get(
            str(normalized_name),
            [],
        )
        if normalized_name
        else []
    )

    id_matches = (
        npcs_by_id.get(
            source_id,
            [],
        )
        if isinstance(source_id, int)
        else []
    )

    name_and_id_matches = [
        npc
        for npc in name_matches
        if npc.get("npc_id")
        == source_id
    ]

    if len(name_and_id_matches) == 1:
        return {
            "status": "linked",
            "kind": "npc",
            "method": "exact_name_and_id",
            "target": _npc_reference(
                name_and_id_matches[0]
            ),
            "group": None,
            "candidate_targets": [],
        }

    if len(name_matches) == 1:
        target = name_matches[0]

        id_consistency = (
            "unavailable"
            if (
                source_id is None
                or target.get("npc_id") is None
            )
            else (
                "agree"
                if source_id
                == target.get("npc_id")
                else "conflict"
            )
        )

        return {
            "status": "linked",
            "kind": "npc",
            "method": "exact_name",
            "id_consistency": (
                id_consistency
            ),
            "target": _npc_reference(
                target
            ),
            "group": None,
            "candidate_targets": [],
        }

    if len(name_matches) > 1:
        id_selected = [
            npc
            for npc in name_matches
            if npc.get("npc_id")
            == source_id
        ]

        if len(id_selected) == 1:
            return {
                "status": "linked",
                "kind": "npc",
                "method": (
                    "exact_name_and_id_disambiguation"
                ),
                "target": _npc_reference(
                    id_selected[0]
                ),
                "group": None,
                "candidate_targets": [
                    _npc_reference(npc)
                    for npc in name_matches
                ],
            }

        return {
            "status": "group",
            "kind": "npc_family",
            "method": "same_name_npc_family",
            "target": None,
            "group": {
                "group_type": "npc_family",
                "name": source_name,
                "explicit_members": [
                    _npc_reference(npc)
                    for npc in name_matches
                ],
            },
            "candidate_targets": [
                _npc_reference(npc)
                for npc in name_matches
            ],
        }

    # ID-only candidates are diagnostic only. They are
    # never applied without a compatible name.
    return {
        "status": "unresolved",
        "kind": "legacy_or_missing_npc",
        "method": "no_safe_npc_match",
        "target": None,
        "group": None,
        "candidate_targets": [
            _npc_reference(npc)
            for npc in id_matches
        ],
    }


def _link_container_or_other_source(
    source_data: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "classified",
        "kind": "container_or_other",
        "method": (
            "source_type_classification"
        ),
        "target": None,
        "group": {
            "group_type": (
                "container_or_other"
            ),
            "name": source_data.get(
                "name"
            ),
            "explicit_members": [],
        },
        "candidate_targets": [],
    }


def _link_drop_source(
    source_data: dict[str, Any],
    *,
    npcs_by_name: dict[
        str,
        list[dict[str, Any]],
    ],
    npcs_by_id: dict[
        int,
        list[dict[str, Any]],
    ],
) -> dict[str, Any]:
    source_type = source_data.get(
        "source_type"
    )

    if source_type == "npc":
        return _link_npc_source(
            source_data,
            npcs_by_name=npcs_by_name,
            npcs_by_id=npcs_by_id,
        )

    if source_type == "container_or_other":
        return (
            _link_container_or_other_source(
                source_data
            )
        )

    return {
        "status": "unresolved",
        "kind": "unknown_source_type",
        "method": "unknown_source_type",
        "target": None,
        "group": None,
        "candidate_targets": [],
    }


def _apply_source_link(
    source_data: dict[str, Any],
    link: dict[str, Any],
) -> None:
    source_data["kind"] = link["kind"]
    source_data["catalog_link"] = link

    target = link.get("target")

    if target is None:
        source_data[
            "resolved_npc_id"
        ] = None

        source_data[
            "npc_source_catalog_id"
        ] = None

        return

    source_data[
        "resolved_npc_id"
    ] = target["npc_id"]

    source_data[
        "npc_source_catalog_id"
    ] = target[
        "source_catalog_id"
    ]


def link_drop_record(
    drop: dict[str, Any],
    *,
    items_by_name: dict[
        str,
        list[dict[str, Any]],
    ],
    npcs_by_name: dict[
        str,
        list[dict[str, Any]],
    ],
    npcs_by_id: dict[
        int,
        list[dict[str, Any]],
    ],
) -> dict[str, Any]:
    linked = copy.deepcopy(drop)

    item_link = _link_drop_item(
        linked["item"],
        items_by_name=items_by_name,
    )

    source_link = _link_drop_source(
        linked["source"],
        npcs_by_name=npcs_by_name,
        npcs_by_id=npcs_by_id,
    )

    _apply_item_link(
        linked["item"],
        item_link,
    )

    _apply_source_link(
        linked["source"],
        source_link,
    )

    item_usable = (
        item_link["status"]
        in {
            "linked",
            "group",
        }
    )

    source_usable = (
        source_link["status"]
        in {
            "linked",
            "group",
            "classified",
        }
    )

    linking_status = (
        "complete"
        if item_usable and source_usable
        else "partial"
    )

    linked["linking"] = {
        "status": linking_status,

        "item_status": (
            item_link["status"]
        ),

        "source_status": (
            source_link["status"]
        ),

        "item_usable": item_usable,
        "source_usable": source_usable,
    }

    return linked


def link_drops_file(
    items_path: str | Path = DEFAULT_ITEMS_PATH,
    npcs_path: str | Path = DEFAULT_NPCS_PATH,
    drops_path: str | Path = DEFAULT_DROPS_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    items_path = Path(items_path)
    npcs_path = Path(npcs_path)
    drops_path = Path(drops_path)
    output_path = Path(output_path)
    report_path = Path(report_path)

    items = _load_jsonl(
        items_path
    )

    npcs = _load_jsonl(
        npcs_path
    )

    drops = _load_jsonl(
        drops_path
    )

    items_by_name = _build_name_index(
        items
    )

    npcs_by_name = _build_name_index(
        npcs
    )

    npcs_by_id = _build_npc_id_index(
        npcs
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

    item_status_counts: Counter[str] = (
        Counter()
    )

    item_method_counts: Counter[str] = (
        Counter()
    )

    item_kind_counts: Counter[str] = (
        Counter()
    )

    source_status_counts: Counter[str] = (
        Counter()
    )

    source_method_counts: Counter[str] = (
        Counter()
    )

    source_kind_counts: Counter[str] = (
        Counter()
    )

    unresolved_item_counts: Counter[
        str
    ] = Counter()

    ambiguous_item_counts: Counter[
        str
    ] = Counter()

    item_group_counts: Counter[
        str
    ] = Counter()

    unresolved_source_counts: Counter[
        str
    ] = Counter()

    source_group_counts: Counter[
        str
    ] = Counter()

    seen_catalog_ids: set[str] = set()

    with temporary_output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        for drop in drops:
            source_catalog_id = drop.get(
                "source_catalog_id"
            )

            if not source_catalog_id:
                raise ValueError(
                    "Drop record is missing "
                    "source_catalog_id."
                )

            if source_catalog_id in seen_catalog_ids:
                raise ValueError(
                    "Duplicate Drop source_catalog_id: "
                    f"{source_catalog_id}"
                )

            seen_catalog_ids.add(
                source_catalog_id
            )

            linked = link_drop_record(
                drop,
                items_by_name=items_by_name,
                npcs_by_name=npcs_by_name,
                npcs_by_id=npcs_by_id,
            )

            total_records += 1

            if (
                linked["linking"]["status"]
                == "complete"
            ):
                complete_records += 1
            else:
                partial_records += 1

            item_link = linked[
                "item"
            ]["catalog_link"]

            source_link = linked[
                "source"
            ]["catalog_link"]

            item_status_counts.update(
                [item_link["status"]]
            )

            item_method_counts.update(
                [item_link["method"]]
            )

            item_kind_counts.update(
                [item_link["kind"]]
            )

            source_status_counts.update(
                [source_link["status"]]
            )

            source_method_counts.update(
                [source_link["method"]]
            )

            source_kind_counts.update(
                [source_link["kind"]]
            )

            item_name = linked[
                "item"
            ]["name"]

            source_name = linked[
                "source"
            ]["name"]

            if (
                item_link["status"]
                == "unresolved"
            ):
                unresolved_item_counts[
                    item_name
                ] += 1

            elif (
                item_link["status"]
                == "ambiguous"
            ):
                ambiguous_item_counts[
                    item_name
                ] += 1

            elif (
                item_link["status"]
                == "group"
            ):
                item_group_counts[
                    item_name
                ] += 1

            if (
                source_link["status"]
                == "unresolved"
            ):
                unresolved_source_counts[
                    source_name
                ] += 1

            elif (
                source_link["status"]
                == "group"
            ):
                source_group_counts[
                    source_name
                ] += 1

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
            "No Drop records were linked."
        )

    temporary_output_path.replace(
        output_path
    )

    usable_item_records = sum(
        count
        for status, count
        in item_status_counts.items()
        if status in {
            "linked",
            "group",
        }
    )

    usable_source_records = sum(
        count
        for status, count
        in source_status_counts.items()
        if status in {
            "linked",
            "group",
            "classified",
        }
    )

    report = {
        "items_path": str(items_path),
        "npcs_path": str(npcs_path),
        "drops_path": str(drops_path),
        "output_path": str(output_path),

        "item_records": len(items),
        "npc_records": len(npcs),
        "drop_records": total_records,

        "complete_records": (
            complete_records
        ),
        "partial_records": partial_records,

        "item_status_counts": dict(
            item_status_counts.most_common()
        ),

        "item_method_counts": dict(
            item_method_counts.most_common()
        ),

        "item_kind_counts": dict(
            item_kind_counts.most_common()
        ),

        "source_status_counts": dict(
            source_status_counts.most_common()
        ),

        "source_method_counts": dict(
            source_method_counts.most_common()
        ),

        "source_kind_counts": dict(
            source_kind_counts.most_common()
        ),

        "item_usable_coverage_percent": (
            round(
                100.0
                * usable_item_records
                / total_records,
                4,
            )
        ),

        "source_usable_coverage_percent": (
            round(
                100.0
                * usable_source_records
                / total_records,
                4,
            )
        ),

        "complete_record_coverage_percent": (
            round(
                100.0
                * complete_records
                / total_records,
                4,
            )
        ),

        "unresolved_item_counts": dict(
            unresolved_item_counts.most_common()
        ),

        "ambiguous_item_counts": dict(
            ambiguous_item_counts.most_common()
        ),

        "item_group_counts": dict(
            item_group_counts.most_common()
        ),

        "unresolved_source_counts": dict(
            unresolved_source_counts.most_common()
        ),

        "source_group_counts": dict(
            source_group_counts.most_common()
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
