from __future__ import annotations

import json
import shutil
import sqlite3
import zlib
from dataclasses import dataclass
from pathlib import Path

import pytest

from src.knowledge.pipeline import build_terraria_knowledge


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CATALOG_ROOT = PROJECT_ROOT / "data" / "terraria" / "catalog"


@dataclass(frozen=True)
class BuiltCatalog:
    root: Path
    database_path: Path
    build_report: dict


@pytest.fixture(scope="session")
def built_catalog(tmp_path_factory: pytest.TempPathFactory) -> BuiltCatalog:
    catalog_root = tmp_path_factory.mktemp("terraria") / "catalog"
    cleaned_root = catalog_root / "cleaned"
    cleaned_root.mkdir(parents=True)

    for filename in ("Items.jsonl", "NPCs.jsonl", "Recipes.jsonl", "Drops.jsonl"):
        shutil.copy2(
            SOURCE_CATALOG_ROOT / "cleaned" / filename,
            cleaned_root / filename,
        )

    shutil.copy2(
        SOURCE_CATALOG_ROOT / "snapshot_manifest.json",
        catalog_root / "snapshot_manifest.json",
    )

    report = build_terraria_knowledge(
        catalog_root=catalog_root,
        strict_snapshot=True,
        verbose=False,
    )
    return BuiltCatalog(
        root=catalog_root,
        database_path=catalog_root / "terraria_query.sqlite3",
        build_report=report,
    )


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_build_report_passed(built_catalog: BuiltCatalog) -> None:
    report = built_catalog.build_report

    assert report["status"] == "passed"
    assert report["strict_snapshot"] is True
    assert report["summary"]["linked_recipe_records"] == 3409
    assert report["summary"]["linked_drop_records"] == 3144
    assert report["summary"]["resolved_references"] == 14353
    assert report["summary"]["fts_enabled"] is True
    assert report["summary"]["database_size_bytes"] > 0
    assert len(report["summary"]["database_sha256"]) == 64
    assert len(report["snapshot_manifest_sha256"]) == 64


def test_catalog_reference_integrity(built_catalog: BuiltCatalog) -> None:
    report = load_json(
        built_catalog.root / "linked" / "catalog_integrity_report.json"
    )

    assert report["status"] == "passed"
    assert report["snapshot_expectations_validated"] is True

    references = report["reference_integrity"]
    assert references["resolved_references_total"] == 14353
    assert references["dangling_item_references"] == 0
    assert references["dangling_npc_references"] == 0
    assert references["mismatched_item_references"] == 0
    assert references["mismatched_npc_references"] == 0


def test_sqlite_integrity_and_counts(built_catalog: BuiltCatalog) -> None:
    connection = sqlite3.connect(built_catalog.database_path)
    connection.row_factory = sqlite3.Row

    try:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []

        expected_counts = {
            "items": 6283,
            "npcs": 770,
            "recipes": 3409,
            "recipe_variants": 4221,
            "recipe_stations": 4221,
            "recipe_ingredients": 6959,
            "drops": 3144,
        }
        actual_counts = {
            table: connection.execute(
                f"SELECT COUNT(*) FROM {table}"  # table names are internal constants
            ).fetchone()[0]
            for table in expected_counts
        }
        assert actual_counts == expected_counts
    finally:
        connection.close()


def test_query_store_core_queries(built_catalog: BuiltCatalog) -> None:
    from src.knowledge.terraria_query_store import TerrariaQueryStore

    with TerrariaQueryStore(built_catalog.database_path) as store:
        terra_blade = store.get_item("Terra Blade", include_record=False)
        assert terra_blade["status"] == "found"
        assert terra_blade["match"]["item_id"] == 757

        seaweed = store.get_item("Seaweed", include_record=False)
        assert seaweed["status"] == "ambiguous"
        assert {match["item_id"] for match in seaweed["matches"]} == {753, 2338}

        pet_seaweed = store.get_item(
            "Seaweed",
            item_id=753,
            include_record=False,
        )
        assert pet_seaweed["status"] == "found"
        assert pet_seaweed["match"]["internal_name"] == "Seaweed"

        armored_skeleton = store.get_npc(
            "Armored Skeleton",
            npc_id=77,
            include_record=False,
        )
        assert armored_skeleton["status"] == "found"
        assert armored_skeleton["match"]["npc_id"] == 77

        # Punctuation-insensitive FTS should find the canonical recipe.
        assert any(
            row["result_name"] == "Night's Edge"
            for row in store.search_recipes("Night Edge", limit=10)
        )


def test_fact_service_item_recipe_and_drop(built_catalog: BuiltCatalog) -> None:
    from src.knowledge.terraria_fact_service import TerrariaFactService

    with TerrariaFactService(built_catalog.database_path) as service:
        item = service.item("Terra Blade")
        assert item["status"] == "found"
        assert item["facts"]["combat"]["damage"] == 85
        assert item["facts"]["inventory"]["rarity"]["primary"] == 8
        assert item["facts"]["economy"]["sell"]["primary_copper"] == 200000

        recipe = service.recipe("Night's Edge", preferred_only=True)
        assert recipe["status"] == "found"
        assert recipe["facts"]["variant_count"] == 2
        assert recipe["facts"]["linking_status"] == "complete"
        assert recipe["facts"]["record_linking_status"] == "partial"
        assert recipe["warnings"] == []

        ingredients = {
            ingredient["name"]
            for variant in recipe["facts"]["variants"]
            for ingredient in variant["ingredients"]
        }
        assert "Volcano" in ingredients
        assert "Fiery Greatsword" not in ingredients

        drop = service.drops_for_item(
            "Beam Sword",
            mode="normal",
            include_partial=False,
        )
        assert drop["status"] == "found"
        drop_fact = drop["facts"]["drops"][0]
        assert drop_fact["source_name"] == "Armored Skeleton"
        assert drop_fact["npc_id"] == 77
        assert drop_fact["chance"]["display"] == "0.67%"


def test_reverse_recipe_and_disambiguated_drop_queries(
    built_catalog: BuiltCatalog,
) -> None:
    from src.knowledge.terraria_fact_service import TerrariaFactService

    with TerrariaFactService(built_catalog.database_path) as service:
        result = service.recipes_using_item("Terra Blade", preferred_only=True)
        assert result["status"] == "found"
        assert any(
            recipe["result_name"] == "Zenith"
            for recipe in result["facts"]["recipes"]
        )

        seaweed = service.drops_for_item(
            "Seaweed",
            item_id=753,
            mode="normal",
            include_partial=False,
        )
        assert seaweed["status"] == "found"
        assert seaweed["facts"]["drop_count"] == 3


def test_dynamic_integrity_audit_without_snapshot(built_catalog: BuiltCatalog) -> None:
    """The audit remains usable after a future catalog refresh."""

    from src.knowledge.linking.catalog_integrity import audit_catalog_integrity

    report = audit_catalog_integrity(
        items_path=built_catalog.root / "cleaned" / "Items.jsonl",
        npcs_path=built_catalog.root / "cleaned" / "NPCs.jsonl",
        cleaned_recipes_path=built_catalog.root / "cleaned" / "Recipes.jsonl",
        linked_recipes_path=built_catalog.root / "linked" / "Recipes.jsonl",
        cleaned_drops_path=built_catalog.root / "cleaned" / "Drops.jsonl",
        linked_drops_path=built_catalog.root / "linked" / "Drops.jsonl",
        report_path=built_catalog.root / "linked" / "dynamic_integrity.json",
        expected_summary=None,
    )
    assert report["status"] == "passed"
    assert report["snapshot_expectations_validated"] is False


def test_database_relationships_json_and_ranges(built_catalog: BuiltCatalog) -> None:
    connection = sqlite3.connect(built_catalog.database_path)
    connection.row_factory = sqlite3.Row
    try:
        scalar_checks = {
            "orphan_variants": """
                SELECT COUNT(*) FROM recipe_variants v
                LEFT JOIN recipes r ON r.source_catalog_id = v.recipe_catalog_id
                WHERE r.source_catalog_id IS NULL
            """,
            "orphan_ingredients": """
                SELECT COUNT(*) FROM recipe_ingredients i
                LEFT JOIN recipe_variants v ON v.variant_id = i.variant_id
                WHERE v.variant_id IS NULL
            """,
            "invalid_linked_ingredients": """
                SELECT COUNT(*) FROM recipe_ingredients
                WHERE link_status = 'linked' AND item_catalog_id IS NULL
            """,
            "invalid_linked_drop_items": """
                SELECT COUNT(*) FROM drops
                WHERE item_link_status = 'linked' AND item_catalog_id IS NULL
            """,
            "invalid_linked_drop_sources": """
                SELECT COUNT(*) FROM drops
                WHERE source_link_status = 'linked' AND npc_catalog_id IS NULL
            """,
            "invalid_drop_ranges": """
                SELECT COUNT(*) FROM drops
                WHERE quantity_minimum < 0
                   OR quantity_maximum <= 0
                   OR quantity_minimum > quantity_maximum
            """,
        }
        for query in scalar_checks.values():
            assert connection.execute(query).fetchone()[0] == 0

        for table, column in (
            ("drops", "chance_by_mode_json"),
            ("drops", "quantity_by_mode_json"),
        ):
            assert connection.execute(
                f"SELECT COUNT(*) FROM {table} WHERE json_valid({column}) = 0"
            ).fetchone()[0] == 0

        # Full record payloads are compressed BLOBs in schema 1.1.
        for table in (
            "items",
            "npcs",
            "recipes",
            "recipe_variants",
            "recipe_ingredients",
            "drops",
        ):
            assert connection.execute(
                f"SELECT COUNT(*) FROM {table} "
                "WHERE typeof(record_json) != 'blob'"
            ).fetchone()[0] == 0
            samples = connection.execute(
                f"SELECT record_json FROM {table} LIMIT 5"
            ).fetchall()
            for sample in samples:
                payload = json.loads(zlib.decompress(sample[0]).decode("utf-8"))
                assert isinstance(payload, dict)

        zero_minimum = connection.execute(
            """
            SELECT item_name, source_name, quantity_minimum, quantity_maximum
            FROM drops WHERE quantity_minimum = 0
            """
        ).fetchall()
        assert [tuple(row) for row in zero_minimum] == [
            ("Shadow Scale", "Eater of Worlds", 0, 134)
        ]
    finally:
        connection.close()
