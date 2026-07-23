"""Reusable Terraria knowledge-catalog build pipeline.

The tracked cleaned JSONL snapshot is the build input.  This module links
entities, audits referential integrity, and creates the read-only SQLite query
database.  The command-line entry point in ``scripts/`` is intentionally thin
so tests and other applications can call the same implementation directly.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.utils.paths import TERRARIA_CATALOG_ROOT, portable_path

from .catalog_database_builder import build_query_database
from .linking.catalog_integrity import audit_catalog_integrity
from .linking.drop_entity_linker import link_drops_file
from .linking.recipe_item_linker import link_recipes_file


@dataclass(frozen=True)
class CatalogBuildPaths:
    """All input and output paths for one catalog build."""

    catalog_root: Path
    database_path: Path
    pipeline_report_path: Path

    @classmethod
    def from_root(
        cls,
        catalog_root: str | Path = TERRARIA_CATALOG_ROOT,
        *,
        database_path: str | Path | None = None,
        pipeline_report_path: str | Path | None = None,
    ) -> "CatalogBuildPaths":
        root = Path(catalog_root).expanduser().resolve()
        return cls(
            catalog_root=root,
            database_path=(
                Path(database_path).expanduser().resolve()
                if database_path is not None
                else root / "terraria_query.sqlite3"
            ),
            pipeline_report_path=(
                Path(pipeline_report_path).expanduser().resolve()
                if pipeline_report_path is not None
                else root / "terraria_build_report.json"
            ),
        )

    @property
    def cleaned_root(self) -> Path:
        return self.catalog_root / "cleaned"

    @property
    def linked_root(self) -> Path:
        return self.catalog_root / "linked"

    @property
    def items_path(self) -> Path:
        return self.cleaned_root / "Items.jsonl"

    @property
    def npcs_path(self) -> Path:
        return self.cleaned_root / "NPCs.jsonl"

    @property
    def recipes_path(self) -> Path:
        return self.cleaned_root / "Recipes.jsonl"

    @property
    def drops_path(self) -> Path:
        return self.cleaned_root / "Drops.jsonl"

    @property
    def linked_recipes_path(self) -> Path:
        return self.linked_root / "Recipes.jsonl"

    @property
    def linked_drops_path(self) -> Path:
        return self.linked_root / "Drops.jsonl"

    @property
    def recipe_report_path(self) -> Path:
        return self.linked_root / "Recipes_link_report.json"

    @property
    def drop_report_path(self) -> Path:
        return self.linked_root / "Drops_link_report.json"

    @property
    def integrity_report_path(self) -> Path:
        return self.linked_root / "catalog_integrity_report.json"

    @property
    def database_report_path(self) -> Path:
        return self.catalog_root / "terraria_query_report.json"

    @property
    def snapshot_manifest_path(self) -> Path:
        return self.catalog_root / "snapshot_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def _require_nonempty_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required input file not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Required input file is empty: {path}")


def _assert_expected_subset(
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    context: str,
) -> None:
    for key, expected_value in expected.items():
        if key not in actual:
            raise AssertionError(f"Missing expected field: {context}.{key}")
        actual_value = actual[key]
        if isinstance(expected_value, dict):
            if not isinstance(actual_value, dict):
                raise AssertionError(f"Expected mapping at {context}.{key}.")
            _assert_expected_subset(
                actual_value,
                expected_value,
                context=f"{context}.{key}",
            )
        elif actual_value != expected_value:
            raise AssertionError(
                f"Snapshot mismatch at {context}.{key}: "
                f"expected {expected_value!r}, got {actual_value!r}."
            )


def _run_stage(
    name: str,
    function: Callable[..., dict[str, Any]],
    *,
    verbose: bool,
    **kwargs: Any,
) -> tuple[dict[str, Any], float]:
    if verbose:
        print(f"\n{'=' * 100}\n{name}\n{'=' * 100}")

    started = time.perf_counter()
    report = function(**kwargs)
    elapsed = time.perf_counter() - started

    if verbose:
        print(f"{name} completed in {elapsed:.2f} seconds.")
        print(json.dumps(report, ensure_ascii=False, indent=2))

    return report, elapsed


def _load_manifest(
    paths: CatalogBuildPaths,
    *,
    strict_snapshot: bool,
) -> dict[str, Any] | None:
    if not paths.snapshot_manifest_path.exists():
        if strict_snapshot:
            raise FileNotFoundError(
                "Strict snapshot validation requested, but the manifest is "
                f"missing: {paths.snapshot_manifest_path}"
            )
        return None

    manifest = _load_json(paths.snapshot_manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported Terraria snapshot manifest schema.")
    return manifest


def _validate_cleaned_snapshot(
    paths: CatalogBuildPaths,
    manifest: dict[str, Any],
) -> None:
    files = {
        "items": paths.items_path,
        "npcs": paths.npcs_path,
        "recipes": paths.recipes_path,
        "drops": paths.drops_path,
    }

    expected_hashes = manifest.get("cleaned_sha256", {})
    expected_counts = manifest.get("expected", {}).get("cleaned_counts", {})

    for name, path in files.items():
        expected_hash = expected_hashes.get(name)
        if expected_hash and _sha256(path) != expected_hash:
            raise AssertionError(f"Cleaned snapshot hash mismatch: {name}")

        expected_count = expected_counts.get(name)
        if expected_count is not None and _count_jsonl(path) != expected_count:
            raise AssertionError(f"Cleaned snapshot count mismatch: {name}")


def build_terraria_knowledge(
    *,
    catalog_root: str | Path = TERRARIA_CATALOG_ROOT,
    database_path: str | Path | None = None,
    pipeline_report_path: str | Path | None = None,
    strict_snapshot: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Build linked data, audit it, and create the query database."""

    paths = CatalogBuildPaths.from_root(
        catalog_root,
        database_path=database_path,
        pipeline_report_path=pipeline_report_path,
    )

    input_paths = {
        "items": paths.items_path,
        "npcs": paths.npcs_path,
        "recipes": paths.recipes_path,
        "drops": paths.drops_path,
    }
    for path in input_paths.values():
        _require_nonempty_file(path)

    paths.linked_root.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(paths, strict_snapshot=strict_snapshot)
    if strict_snapshot and manifest is not None:
        _validate_cleaned_snapshot(paths, manifest)

    expected = manifest.get("expected", {}) if manifest else {}
    pipeline_started = time.perf_counter()

    recipe_report, recipe_seconds = _run_stage(
        "1/4 LINK RECIPES",
        link_recipes_file,
        verbose=verbose,
        items_path=paths.items_path,
        recipes_path=paths.recipes_path,
        output_path=paths.linked_recipes_path,
        report_path=paths.recipe_report_path,
    )
    if strict_snapshot:
        _assert_expected_subset(
            recipe_report,
            expected.get("recipe_link_report", {}),
            context="recipe_link_report",
        )

    drop_report, drop_seconds = _run_stage(
        "2/4 LINK DROPS",
        link_drops_file,
        verbose=verbose,
        items_path=paths.items_path,
        npcs_path=paths.npcs_path,
        drops_path=paths.drops_path,
        output_path=paths.linked_drops_path,
        report_path=paths.drop_report_path,
    )
    if strict_snapshot:
        _assert_expected_subset(
            drop_report,
            expected.get("drop_link_report", {}),
            context="drop_link_report",
        )

    integrity_report, integrity_seconds = _run_stage(
        "3/4 AUDIT CATALOG INTEGRITY",
        audit_catalog_integrity,
        verbose=verbose,
        items_path=paths.items_path,
        npcs_path=paths.npcs_path,
        cleaned_recipes_path=paths.recipes_path,
        linked_recipes_path=paths.linked_recipes_path,
        cleaned_drops_path=paths.drops_path,
        linked_drops_path=paths.linked_drops_path,
        report_path=paths.integrity_report_path,
        expected_summary=(
            expected.get("integrity_report") if strict_snapshot else None
        ),
    )

    database_report, database_seconds = _run_stage(
        "4/4 BUILD QUERY DATABASE",
        build_query_database,
        verbose=verbose,
        items_path=paths.items_path,
        npcs_path=paths.npcs_path,
        recipes_path=paths.linked_recipes_path,
        drops_path=paths.linked_drops_path,
        database_path=paths.database_path,
        report_path=paths.database_report_path,
    )
    if strict_snapshot:
        _assert_expected_subset(
            database_report.get("table_counts", {}),
            expected.get("database_counts", {}),
            context="database_table_counts",
        )

    total_seconds = time.perf_counter() - pipeline_started
    manifest_sha = (
        _sha256(paths.snapshot_manifest_path)
        if manifest is not None
        else None
    )

    pipeline_report = {
        "status": "passed",
        "catalog_root": portable_path(paths.catalog_root),
        "database_path": portable_path(paths.database_path),
        "strict_snapshot": strict_snapshot,
        "snapshot_manifest": (
            portable_path(paths.snapshot_manifest_path) if manifest is not None else None
        ),
        "snapshot_manifest_sha256": manifest_sha,
        "inputs": {name: portable_path(path) for name, path in input_paths.items()},
        "outputs": {
            "linked_recipes": portable_path(paths.linked_recipes_path),
            "linked_drops": portable_path(paths.linked_drops_path),
            "integrity_report": portable_path(paths.integrity_report_path),
            "query_database": portable_path(paths.database_path),
            "query_database_report": portable_path(paths.database_report_path),
        },
        "timings_seconds": {
            "link_recipes": round(recipe_seconds, 4),
            "link_drops": round(drop_seconds, 4),
            "integrity_audit": round(integrity_seconds, 4),
            "query_database": round(database_seconds, 4),
            "total": round(total_seconds, 4),
        },
        "summary": {
            "linked_recipe_records": recipe_report["recipe_records"],
            "linked_drop_records": drop_report["drop_records"],
            "resolved_references": integrity_report["reference_integrity"][
                "resolved_references_total"
            ],
            "database_size_bytes": database_report["database_size_bytes"],
            "database_sha256": database_report["database_sha256"],
            "fts_enabled": database_report["fts_enabled"],
        },
    }

    _write_json(paths.pipeline_report_path, pipeline_report)

    if verbose:
        print(f"\n{'=' * 100}\nTERRARIA KNOWLEDGE BUILD PASSED\n{'=' * 100}")
        print(json.dumps(pipeline_report, ensure_ascii=False, indent=2))

    return pipeline_report
