from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

from src.knowledge.pipeline import build_terraria_knowledge


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CATALOG_ROOT = PROJECT_ROOT / "data" / "terraria" / "catalog"


@dataclass(frozen=True)
class AssistantCatalog:
    root: Path
    database_path: Path


@pytest.fixture(scope="session")
def assistant_catalog(tmp_path_factory: pytest.TempPathFactory) -> AssistantCatalog:
    catalog_root = tmp_path_factory.mktemp("assistant_terraria") / "catalog"
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

    build_terraria_knowledge(
        catalog_root=catalog_root,
        strict_snapshot=True,
        verbose=False,
    )
    return AssistantCatalog(
        root=catalog_root,
        database_path=catalog_root / "terraria_query.sqlite3",
    )
