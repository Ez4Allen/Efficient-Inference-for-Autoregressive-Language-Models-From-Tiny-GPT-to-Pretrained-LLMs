
from __future__ import annotations

import json
import sqlite3

from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

CATALOG_ROOT = (
    PROJECT_ROOT
    / "data/terraria/catalog"
)

DATABASE_PATH = (
    CATALOG_ROOT
    / "terraria_query.sqlite3"
)

BUILD_REPORT_PATH = (
    CATALOG_ROOT
    / "terraria_build_report.json"
)

INTEGRITY_REPORT_PATH = (
    CATALOG_ROOT
    / "linked/catalog_integrity_report.json"
)


def load_json(
    path: Path,
) -> dict:
    assert path.exists(), (
        f"Missing required file: {path}"
    )

    return json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )


def test_build_report_passed() -> None:
    report = load_json(
        BUILD_REPORT_PATH
    )

    assert report["status"] == "passed"

    assert report["summary"] == {
        "linked_recipe_records": 3409,
        "linked_drop_records": 3144,
        "resolved_references": 14353,
        "database_size_bytes": 94789632,
        "database_sha256": (
            "5cc4e650e9a4014d6395f148984e1d8f"
            "463f279f3a26bc8da9367efbf37f658e"
        ),
        "fts_enabled": True,
    }


def test_catalog_reference_integrity() -> None:
    report = load_json(
        INTEGRITY_REPORT_PATH
    )

    assert report["status"] == "passed"

    reference_integrity = report[
        "reference_integrity"
    ]

    assert reference_integrity[
        "resolved_references_total"
    ] == 14353

    assert reference_integrity[
        "dangling_item_references"
    ] == 0

    assert reference_integrity[
        "dangling_npc_references"
    ] == 0

    assert reference_integrity[
        "mismatched_item_references"
    ] == 0

    assert reference_integrity[
        "mismatched_npc_references"
    ] == 0


def test_sqlite_integrity_and_counts() -> None:
    assert DATABASE_PATH.exists()

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    try:
        integrity_result = (
            connection.execute(
                "PRAGMA integrity_check"
            ).fetchone()
        )

        assert integrity_result is not None
        assert integrity_result[0] == "ok"

        foreign_key_errors = (
            connection.execute(
                "PRAGMA foreign_key_check"
            ).fetchall()
        )

        assert foreign_key_errors == []

        expected_counts = {
            "items": 6283,
            "npcs": 770,
            "recipes": 3409,
            "recipe_variants": 4221,
            "recipe_stations": 4221,
            "recipe_ingredients": 6959,
            "drops": 3144,
        }

        actual_counts = {}

        for table_name in expected_counts:
            row = connection.execute(
                f"""
                SELECT COUNT(*) AS count
                FROM {table_name}
                """
            ).fetchone()

            assert row is not None

            actual_counts[
                table_name
            ] = int(row["count"])

        assert actual_counts == (
            expected_counts
        )

    finally:
        connection.close()


def test_query_store_core_queries() -> None:
    from src.knowledge.terraria_query_store import (
        TerrariaQueryStore,
    )

    with TerrariaQueryStore(
        DATABASE_PATH
    ) as store:
        terra_blade = store.get_item(
            "Terra Blade",
            include_record=False,
        )

        assert terra_blade[
            "status"
        ] == "found"

        assert terra_blade[
            "match"
        ]["item_id"] == 757

        seaweed = store.get_item(
            "Seaweed",
            include_record=False,
        )

        assert seaweed[
            "status"
        ] == "ambiguous"

        assert {
            match["item_id"]
            for match
            in seaweed["matches"]
        } == {
            753,
            2338,
        }

        armored_skeleton = (
            store.get_npc(
                "Armored Skeleton",
                npc_id=77,
                include_record=False,
            )
        )

        assert armored_skeleton[
            "status"
        ] == "found"

        assert armored_skeleton[
            "match"
        ]["npc_id"] == 77


def test_fact_service_recipe_and_drop() -> None:
    from src.knowledge.terraria_fact_service import (
        TerrariaFactService,
    )

    with TerrariaFactService(
        DATABASE_PATH
    ) as service:
        recipe = service.recipe(
            "Night's Edge",
            preferred_only=True,
        )

        assert recipe[
            "status"
        ] == "found"

        assert recipe["facts"][
            "variant_count"
        ] == 2

        ingredients = {
            ingredient["name"]
            for variant
            in recipe["facts"]["variants"]
            for ingredient
            in variant["ingredients"]
        }

        assert "Volcano" in ingredients

        assert (
            "Fiery Greatsword"
            not in ingredients
        )

        drop = service.drops_for_item(
            "Beam Sword",
            mode="normal",
            include_partial=False,
        )

        assert drop[
            "status"
        ] == "found"

        assert drop["facts"][
            "drop_count"
        ] == 1

        drop_fact = drop[
            "facts"
        ]["drops"][0]

        assert drop_fact[
            "source_name"
        ] == "Armored Skeleton"

        assert drop_fact[
            "npc_id"
        ] == 77

        assert drop_fact[
            "chance"
        ]["display"] == "0.67%"


def test_reverse_recipe_query() -> None:
    from src.knowledge.terraria_fact_service import (
        TerrariaFactService,
    )

    with TerrariaFactService(
        DATABASE_PATH
    ) as service:
        result = (
            service.recipes_using_item(
                "Terra Blade",
                preferred_only=True,
            )
        )

        assert result[
            "status"
        ] == "found"

        assert any(
            recipe["result_name"]
            == "Zenith"
            for recipe
            in result["facts"]["recipes"]
        )
