"""Read-only lookup over the compact Stardew Valley fact database."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from .database_builder import DEFAULT_DATABASE_PATH
from .normalizers import normalize_name


class StardewQueryStore:
    def __init__(self, database_path: str | Path = DEFAULT_DATABASE_PATH) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        if not self.database_path.exists():
            raise FileNotFoundError(
                "Stardew database not found. Run `python scripts/build_stardew_knowledge.py`. "
                f"Expected: {self.database_path}"
            )
        uri = self.database_path.as_uri() + "?mode=ro"
        self.connection = sqlite3.connect(uri, uri=True)
        self.connection.row_factory = sqlite3.Row
        self._closed = False

    def __enter__(self) -> "StardewQueryStore":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self.connection.close()
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("StardewQueryStore is closed.")

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        return json.loads(row["record_json"])

    def counts(self) -> dict[str, int]:
        self._ensure_open()
        rows = self.connection.execute(
            "SELECT record_type, COUNT(*) AS count FROM records GROUP BY record_type"
        ).fetchall()
        result = {row["record_type"]: int(row["count"]) for row in rows}
        result["total"] = sum(result.values())
        return result

    def get_entity(
        self,
        name: str,
        *,
        record_type: str | None = None,
    ) -> dict[str, Any]:
        self._ensure_open()
        normalized = normalize_name(name)
        params: list[Any] = [normalized]
        # Acquisition relations deliberately reuse the canonical entity name.
        # Generic entity lookup excludes those relation records so a crop such
        # as Parsnip does not become ambiguous merely because its acquisition
        # sources are also tracked.
        type_clause = "AND r.record_type <> 'acquisition'"
        if record_type is not None:
            type_clause = "AND r.record_type = ?"
            params.append(str(record_type))
        rows = self.connection.execute(
            f"""
            SELECT DISTINCT r.record_json
            FROM aliases AS a
            JOIN records AS r ON r.source_catalog_id = a.source_catalog_id
            WHERE a.normalized_alias = ? {type_clause}
            ORDER BY r.record_type, r.name
            """,
            tuple(params),
        ).fetchall()
        matches = [self._decode(row) for row in rows]
        if not matches:
            return {"status": "not_found", "query": name, "matches": []}
        return {
            "status": "found" if len(matches) == 1 else "ambiguous",
            "query": name,
            "match": matches[0] if len(matches) == 1 else None,
            "matches": matches,
        }

    def get_crop(self, name: str) -> dict[str, Any]:
        return self.get_entity(name, record_type="crop")

    def get_fish(self, name: str) -> dict[str, Any]:
        return self.get_entity(name, record_type="fish")

    def get_villager(self, name: str) -> dict[str, Any]:
        return self.get_entity(name, record_type="villager")

    def get_recipe(self, name: str) -> dict[str, Any]:
        return self.get_entity(name, record_type="recipe")

    def get_acquisition(self, name: str) -> dict[str, Any]:
        return self.get_entity(name, record_type="acquisition")

    def acquisition_sources(self, entity_name: str) -> list[dict[str, Any]]:
        result = self.get_acquisition(entity_name)
        if result.get("status") == "not_found":
            return []
        records = list(result.get("matches") or [])
        sources: list[dict[str, Any]] = []
        for record in records:
            for source in (record.get("facts") or {}).get("sources") or []:
                item = dict(source)
                item["source_catalog_id"] = record.get("source_catalog_id")
                item["provenance"] = record.get("provenance")
                sources.append(item)
        return sources

    def get_bundle(
        self,
        name: str,
        *,
        bundle_mode: str | None = "standard",
    ) -> dict[str, Any]:
        result = self.get_entity(name, record_type="bundle")
        if bundle_mode is None or result.get("status") == "not_found":
            return result
        normalized_mode = str(bundle_mode).strip().casefold()
        matches = [
            record
            for record in result.get("matches") or []
            if str((record.get("facts") or {}).get("bundle_mode", "")).casefold()
            == normalized_mode
        ]
        if not matches:
            return {
                "status": "not_found",
                "query": name,
                "bundle_mode": normalized_mode,
                "matches": [],
            }
        return {
            "status": "found" if len(matches) == 1 else "ambiguous",
            "query": name,
            "bundle_mode": normalized_mode,
            "match": matches[0] if len(matches) == 1 else None,
            "matches": matches,
        }

    def aliases(self) -> list[dict[str, str]]:
        self._ensure_open()
        rows = self.connection.execute(
            """
            SELECT a.alias, a.normalized_alias, r.name, r.record_type, r.source_catalog_id
            FROM aliases AS a
            JOIN records AS r ON r.source_catalog_id = a.source_catalog_id
            WHERE r.record_type <> 'acquisition'
            ORDER BY LENGTH(a.alias) DESC, a.alias,
                     CASE r.record_type
                         WHEN 'crop' THEN 1
                         WHEN 'fish' THEN 2
                         WHEN 'villager' THEN 3
                         WHEN 'recipe' THEN 4
                         WHEN 'bundle' THEN 5
                         ELSE 9
                     END
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def resolve_entity_in_text(self, text: str) -> dict[str, Any] | None:
        folded = str(text).casefold()
        for candidate in self.aliases():
            alias = str(candidate["alias"]).strip()
            if not alias:
                continue
            folded_alias = alias.casefold()
            if re.search(r"[\u4e00-\u9fff]", alias):
                if folded_alias in folded:
                    return candidate
                continue
            # Latin aliases need token boundaries. Without them, short names
            # such as ``Eel`` would incorrectly match ordinary words such as
            # ``feel``. Spaces and punctuation inside multi-word names remain
            # significant and are matched case-insensitively.
            pattern = rf"(?<![A-Za-z0-9]){re.escape(folded_alias)}(?![A-Za-z0-9])"
            if re.search(pattern, folded):
                return candidate
        return None

    def bundles_requiring_item(
        self,
        item_name: str,
        *,
        bundle_mode: str = "standard",
    ) -> list[dict[str, Any]]:
        self._ensure_open()
        needle = normalize_name(item_name)
        rows = self.connection.execute(
            "SELECT record_json FROM records WHERE record_type='bundle' ORDER BY name"
        ).fetchall()
        results = []
        for row in rows:
            record = self._decode(row)
            facts = record.get("facts") or {}
            if facts.get("bundle_mode") != bundle_mode:
                continue
            for requirement in facts.get("requirements") or []:
                if normalize_name(requirement.get("item_name", "")) == needle:
                    results.append(record)
                    break
        return results

    def recipes_using_item(self, item_name: str) -> list[dict[str, Any]]:
        self._ensure_open()
        needle = normalize_name(item_name)
        rows = self.connection.execute(
            "SELECT record_json FROM records WHERE record_type='recipe' ORDER BY name"
        ).fetchall()
        results = []
        for row in rows:
            record = self._decode(row)
            for ingredient in (record.get("facts") or {}).get("ingredients") or []:
                if normalize_name(ingredient.get("item_name", "")) == needle:
                    results.append(record)
                    break
        return results

    @staticmethod
    def _fts_query(query: str) -> str:
        tokens = [token.replace('"', '""') for token in str(query).split() if token]
        return " OR ".join(f'"{token}"' for token in tokens)

    def search(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        self._ensure_open()
        limit = max(1, min(int(limit), 100))
        exact = self.get_entity(query)
        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        for record in exact.get("matches") or []:
            record = dict(record)
            record["match_method"] = "exact"
            results.append(record)
            seen.add(record["source_catalog_id"])
        if not str(query).strip():
            return results[:limit]
        try:
            rows = self.connection.execute(
                """
                SELECT r.record_json, bm25(records_fts) AS rank
                FROM records_fts
                JOIN records AS r ON r.source_catalog_id = records_fts.source_catalog_id
                WHERE records_fts MATCH ?
                ORDER BY rank, r.name
                LIMIT ?
                """,
                (self._fts_query(query), limit * 2),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        for row in rows:
            record = self._decode(row)
            if record["source_catalog_id"] in seen:
                continue
            record["match_method"] = "fts"
            results.append(record)
            seen.add(record["source_catalog_id"])
            if len(results) >= limit:
                break
        return results
