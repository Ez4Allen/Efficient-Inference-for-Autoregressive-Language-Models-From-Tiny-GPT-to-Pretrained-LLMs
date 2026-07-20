
from __future__ import annotations

import argparse
import json
import sys
import time

from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.knowledge.catalog_database_builder import (
    build_query_database,
)

from src.knowledge.linking.catalog_integrity import (
    audit_catalog_integrity,
)

from src.knowledge.linking.drop_entity_linker import (
    link_drops_file,
)

from src.knowledge.linking.recipe_item_linker import (
    link_recipes_file,
)


CATALOG_ROOT = (
    PROJECT_ROOT
    / "data/terraria/catalog"
)

CLEANED_ROOT = (
    CATALOG_ROOT
    / "cleaned"
)

LINKED_ROOT = (
    CATALOG_ROOT
    / "linked"
)

DEFAULT_ITEMS_PATH = (
    CLEANED_ROOT
    / "Items.jsonl"
)

DEFAULT_NPCS_PATH = (
    CLEANED_ROOT
    / "NPCs.jsonl"
)

DEFAULT_RECIPES_PATH = (
    CLEANED_ROOT
    / "Recipes.jsonl"
)

DEFAULT_DROPS_PATH = (
    CLEANED_ROOT
    / "Drops.jsonl"
)

DEFAULT_LINKED_RECIPES_PATH = (
    LINKED_ROOT
    / "Recipes.jsonl"
)

DEFAULT_LINKED_DROPS_PATH = (
    LINKED_ROOT
    / "Drops.jsonl"
)

DEFAULT_RECIPE_REPORT_PATH = (
    LINKED_ROOT
    / "Recipes_link_report.json"
)

DEFAULT_DROP_REPORT_PATH = (
    LINKED_ROOT
    / "Drops_link_report.json"
)

DEFAULT_INTEGRITY_REPORT_PATH = (
    LINKED_ROOT
    / "catalog_integrity_report.json"
)

DEFAULT_DATABASE_PATH = (
    CATALOG_ROOT
    / "terraria_query.sqlite3"
)

DEFAULT_DATABASE_REPORT_PATH = (
    CATALOG_ROOT
    / "terraria_query_report.json"
)

DEFAULT_PIPELINE_REPORT_PATH = (
    CATALOG_ROOT
    / "terraria_build_report.json"
)


EXPECTED_COUNTS = {
    "items": 6283,
    "npcs": 770,
    "recipes": 3409,
    "drops": 3144,
    "recipe_variants": 4221,
    "recipe_ingredients": 6959,
}


def _require_file(
    path: Path,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file not found: "
            f"{path}"
        )

    if not path.is_file():
        raise FileNotFoundError(
            f"Required input is not a file: "
            f"{path}"
        )

    if path.stat().st_size == 0:
        raise ValueError(
            f"Required input file is empty: "
            f"{path}"
        )


def _write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(path)


def _run_stage(
    name: str,
    function,
    **kwargs,
) -> tuple[
    dict[str, Any],
    float,
]:
    print()
    print("=" * 110)
    print(name)
    print("=" * 110)

    start = time.perf_counter()

    result = function(
        **kwargs
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"{name} completed in "
        f"{elapsed:.2f} seconds."
    )

    if isinstance(result, dict):
        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            )
        )

    return result, elapsed


def _validate_recipe_report(
    report: dict[str, Any],
) -> None:
    if report.get(
        "recipe_records"
    ) != EXPECTED_COUNTS["recipes"]:
        raise AssertionError(
            "Unexpected linked recipe count."
        )

    if report.get(
        "complete_records"
    ) != 3304:
        raise AssertionError(
            "Unexpected complete recipe count."
        )

    if report.get(
        "partial_records"
    ) != 105:
        raise AssertionError(
            "Unexpected partial recipe count."
        )

    if report.get(
        "total_variants"
    ) != EXPECTED_COUNTS[
        "recipe_variants"
    ]:
        raise AssertionError(
            "Unexpected recipe variant count."
        )

    if report.get(
        "total_ingredient_entries"
    ) != EXPECTED_COUNTS[
        "recipe_ingredients"
    ]:
        raise AssertionError(
            "Unexpected recipe ingredient count."
        )


def _validate_drop_report(
    report: dict[str, Any],
) -> None:
    if report.get(
        "drop_records"
    ) != EXPECTED_COUNTS["drops"]:
        raise AssertionError(
            "Unexpected linked drop count."
        )

    if report.get(
        "complete_records"
    ) != 3134:
        raise AssertionError(
            "Unexpected complete drop count."
        )

    if report.get(
        "partial_records"
    ) != 10:
        raise AssertionError(
            "Unexpected partial drop count."
        )


def _validate_integrity_report(
    report: dict[str, Any],
) -> None:
    if report.get("status") != "passed":
        raise AssertionError(
            "Catalog integrity audit did not pass."
        )

    reference_integrity = report[
        "reference_integrity"
    ]

    if reference_integrity[
        "resolved_references_total"
    ] != 14353:
        raise AssertionError(
            "Unexpected resolved reference count."
        )

    for field in (
        "dangling_item_references",
        "dangling_npc_references",
        "mismatched_item_references",
        "mismatched_npc_references",
    ):
        if reference_integrity[field] != 0:
            raise AssertionError(
                f"Catalog integrity failure: "
                f"{field}={reference_integrity[field]}"
            )


def _validate_database_report(
    report: dict[str, Any],
) -> None:
    if report.get("status") != "passed":
        raise AssertionError(
            "Query database build did not pass."
        )

    expected_input_counts = {
        "items": EXPECTED_COUNTS[
            "items"
        ],
        "npcs": EXPECTED_COUNTS[
            "npcs"
        ],
        "recipes": EXPECTED_COUNTS[
            "recipes"
        ],
        "drops": EXPECTED_COUNTS[
            "drops"
        ],
    }

    if report.get(
        "input_counts"
    ) != expected_input_counts:
        raise AssertionError(
            "Unexpected query database "
            "input counts."
        )

    integrity = report.get(
        "integrity",
        {},
    )

    if integrity.get(
        "sqlite_integrity_check"
    ) != "ok":
        raise AssertionError(
            "SQLite integrity check failed."
        )

    if integrity.get(
        "foreign_key_errors"
    ) != 0:
        raise AssertionError(
            "SQLite foreign-key check failed."
        )


def build_terraria_knowledge(
    *,
    database_path: Path = (
        DEFAULT_DATABASE_PATH
    ),
    pipeline_report_path: Path = (
        DEFAULT_PIPELINE_REPORT_PATH
    ),
) -> dict[str, Any]:
    inputs = {
        "items": DEFAULT_ITEMS_PATH,
        "npcs": DEFAULT_NPCS_PATH,
        "recipes": DEFAULT_RECIPES_PATH,
        "drops": DEFAULT_DROPS_PATH,
    }

    for path in inputs.values():
        _require_file(path)

    LINKED_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    pipeline_start = time.perf_counter()

    recipe_report, recipe_seconds = (
        _run_stage(
            "1/4 LINK RECIPES",
            link_recipes_file,
            items_path=DEFAULT_ITEMS_PATH,
            recipes_path=(
                DEFAULT_RECIPES_PATH
            ),
            output_path=(
                DEFAULT_LINKED_RECIPES_PATH
            ),
            report_path=(
                DEFAULT_RECIPE_REPORT_PATH
            ),
        )
    )

    _validate_recipe_report(
        recipe_report
    )

    drop_report, drop_seconds = (
        _run_stage(
            "2/4 LINK DROPS",
            link_drops_file,
            items_path=DEFAULT_ITEMS_PATH,
            npcs_path=DEFAULT_NPCS_PATH,
            drops_path=DEFAULT_DROPS_PATH,
            output_path=(
                DEFAULT_LINKED_DROPS_PATH
            ),
            report_path=(
                DEFAULT_DROP_REPORT_PATH
            ),
        )
    )

    _validate_drop_report(
        drop_report
    )

    integrity_report, integrity_seconds = (
        _run_stage(
            "3/4 AUDIT CATALOG INTEGRITY",
            audit_catalog_integrity,
            items_path=DEFAULT_ITEMS_PATH,
            npcs_path=DEFAULT_NPCS_PATH,
            cleaned_recipes_path=(
                DEFAULT_RECIPES_PATH
            ),
            linked_recipes_path=(
                DEFAULT_LINKED_RECIPES_PATH
            ),
            cleaned_drops_path=(
                DEFAULT_DROPS_PATH
            ),
            linked_drops_path=(
                DEFAULT_LINKED_DROPS_PATH
            ),
            report_path=(
                DEFAULT_INTEGRITY_REPORT_PATH
            ),
        )
    )

    _validate_integrity_report(
        integrity_report
    )

    database_report, database_seconds = (
        _run_stage(
            "4/4 BUILD QUERY DATABASE",
            build_query_database,
            items_path=DEFAULT_ITEMS_PATH,
            npcs_path=DEFAULT_NPCS_PATH,
            recipes_path=(
                DEFAULT_LINKED_RECIPES_PATH
            ),
            drops_path=(
                DEFAULT_LINKED_DROPS_PATH
            ),
            database_path=database_path,
            report_path=(
                DEFAULT_DATABASE_REPORT_PATH
            ),
        )
    )

    _validate_database_report(
        database_report
    )

    total_seconds = (
        time.perf_counter()
        - pipeline_start
    )

    pipeline_report = {
        "status": "passed",
        "project_root": str(
            PROJECT_ROOT
        ),
        "catalog_root": str(
            CATALOG_ROOT
        ),
        "database_path": str(
            database_path
        ),
        "inputs": {
            key: str(path)
            for key, path
            in inputs.items()
        },
        "outputs": {
            "linked_recipes": str(
                DEFAULT_LINKED_RECIPES_PATH
            ),
            "linked_drops": str(
                DEFAULT_LINKED_DROPS_PATH
            ),
            "integrity_report": str(
                DEFAULT_INTEGRITY_REPORT_PATH
            ),
            "query_database": str(
                database_path
            ),
            "query_database_report": str(
                DEFAULT_DATABASE_REPORT_PATH
            ),
        },
        "timings_seconds": {
            "link_recipes": round(
                recipe_seconds,
                4,
            ),
            "link_drops": round(
                drop_seconds,
                4,
            ),
            "integrity_audit": round(
                integrity_seconds,
                4,
            ),
            "query_database": round(
                database_seconds,
                4,
            ),
            "total": round(
                total_seconds,
                4,
            ),
        },
        "summary": {
            "linked_recipe_records": (
                recipe_report[
                    "recipe_records"
                ]
            ),
            "linked_drop_records": (
                drop_report["drop_records"]
            ),
            "resolved_references": (
                integrity_report[
                    "reference_integrity"
                ][
                    "resolved_references_total"
                ]
            ),
            "database_size_bytes": (
                database_report[
                    "database_size_bytes"
                ]
            ),
            "database_sha256": (
                database_report[
                    "database_sha256"
                ]
            ),
            "fts_enabled": (
                database_report[
                    "fts_enabled"
                ]
            ),
        },
    }

    _write_json(
        pipeline_report_path,
        pipeline_report,
    )

    print()
    print("=" * 110)
    print("TERRARIA KNOWLEDGE BUILD PASSED")
    print("=" * 110)

    print(
        json.dumps(
            pipeline_report,
            ensure_ascii=False,
            indent=2,
        )
    )

    return pipeline_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build linked Terraria catalog data, "
            "run integrity checks and create the "
            "read-only query database."
        )
    )

    parser.add_argument(
        "--database-path",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
        help=(
            "Output SQLite query database path."
        ),
    )

    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_PIPELINE_REPORT_PATH,
        help=(
            "Output pipeline summary report path."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    build_terraria_knowledge(
        database_path=args.database_path,
        pipeline_report_path=(
            args.report_path
        ),
    )


if __name__ == "__main__":
    main()
