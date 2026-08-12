#!/usr/bin/env python3
"""Offline release validation for the tracked GameGuideLM source tree."""

from __future__ import annotations

import argparse
import compileall
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gameguide import __version__
from src.games.stardew.database_builder import build_stardew_database
from src.games.stardew.fact_service import StardewFactService
from src.knowledge.pipeline import build_terraria_knowledge
from src.knowledge.terraria_fact_service import TerrariaFactService
from src.utils.paths import STARDEW_CATALOG_ROOT, TERRARIA_CATALOG_ROOT


def _run_tests() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=PROJECT_ROOT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("The pytest release suite failed.")


def _validate_stardew(temp_root: Path) -> dict[str, object]:
    database = temp_root / "stardew_query.sqlite3"
    report = temp_root / "stardew_build_report.json"
    return build_stardew_database(
        facts_path=STARDEW_CATALOG_ROOT / "cleaned" / "facts.jsonl",
        database_path=database,
        report_path=report,
    )


def _validate_terraria(temp_root: Path) -> dict[str, object]:
    catalog_root = temp_root / "terraria_catalog"
    (catalog_root / "cleaned").mkdir(parents=True, exist_ok=True)
    for name in ("Items.jsonl", "NPCs.jsonl", "Recipes.jsonl", "Drops.jsonl"):
        shutil.copy2(TERRARIA_CATALOG_ROOT / "cleaned" / name, catalog_root / "cleaned" / name)
    shutil.copy2(
        TERRARIA_CATALOG_ROOT / "snapshot_manifest.json",
        catalog_root / "snapshot_manifest.json",
    )
    return build_terraria_knowledge(
        catalog_root=catalog_root,
        database_path=catalog_root / "terraria_query.sqlite3",
        pipeline_report_path=catalog_root / "terraria_build_report.json",
        strict_snapshot=True,
        verbose=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile, test, and rebuild both tracked structured knowledge snapshots."
    )
    parser.add_argument(
        "--skip-pytest",
        action="store_true",
        help="Skip pytest when it has already been run in the same environment.",
    )
    args = parser.parse_args()

    if not compileall.compile_dir(PROJECT_ROOT / "src", quiet=1):
        raise RuntimeError("Python compilation failed under src/.")
    if not compileall.compile_dir(PROJECT_ROOT / "scripts", quiet=1):
        raise RuntimeError("Python compilation failed under scripts/.")

    if not args.skip_pytest:
        _run_tests()

    with tempfile.TemporaryDirectory(prefix="gameguidelm_release_") as temporary:
        temp_root = Path(temporary)
        stardew = _validate_stardew(temp_root)
        stardew_database = temp_root / "stardew_query.sqlite3"
        terraria = _validate_terraria(temp_root)
        terraria_database = temp_root / "terraria_catalog" / "terraria_query.sqlite3"

        with StardewFactService(stardew_database) as service:
            crop_smoke = service.crop_deadline(
                "Parsnip",
                player_state={"season": "spring", "day": 24},
            )
        if crop_smoke["status"] != "found" or not crop_smoke["facts"][
            "can_harvest_before_season_end"
        ]:
            raise AssertionError("Stardew deterministic smoke query failed.")

        with TerrariaFactService(terraria_database) as service:
            recipe_smoke = service.recipe("Night's Edge", preferred_only=True)
        if recipe_smoke["status"] != "found" or recipe_smoke["facts"][
            "variant_count"
        ] != 2:
            raise AssertionError("Terraria deterministic smoke query failed.")

    payload = {
        "status": "passed",
        "project": "GameGuideLM",
        "version": __version__,
        "stardew": {
            "record_count": stardew["record_count"],
            "integrity_check": stardew["integrity_check"],
            "smoke_query": "Parsnip deadline passed",
        },
        "terraria": {
            "linked_recipe_records": terraria["summary"]["linked_recipe_records"],
            "linked_drop_records": terraria["summary"]["linked_drop_records"],
            "resolved_references": terraria["summary"]["resolved_references"],
            "smoke_query": "Night's Edge recipe passed",
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
