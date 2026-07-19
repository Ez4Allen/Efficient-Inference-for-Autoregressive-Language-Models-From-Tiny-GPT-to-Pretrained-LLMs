
from __future__ import annotations

import json
import re
import sqlite3
import unicodedata

from collections.abc import Iterable
from pathlib import Path
from typing import Any


DEFAULT_CATALOG_PATH = Path(
    "/content/llm_project/data/terraria/"
    "catalog/terraria_catalog.sqlite3"
)


RECORD_TYPE_ALIASES = {
    "item": "Items",
    "items": "Items",

    "npc": "NPCs",
    "npcs": "NPCs",
    "boss": "NPCs",

    # “recipe” 默认查询已经合并好的配方实体，
    # 而不是 Cargo 中的单条原始配方行。
    "recipe": "RecipeEntities",
    "recipes": "RecipeEntities",
    "recipe_entity": "RecipeEntities",
    "recipe_entities": "RecipeEntities",
    "grouped_recipe": "RecipeEntities",

    # 需要检查 Cargo 原始配方行时使用。
    "raw_recipe": "Recipes",
    "recipe_row": "Recipes",
    "recipe_rows": "Recipes",

    "drop": "Drops",
    "drops": "Drops",

    # 也允许直接使用数据库中的正式名称。
    "Items": "Items",
    "NPCs": "NPCs",
    "Recipes": "Recipes",
    "Drops": "Drops",
    "RecipeEntities": "RecipeEntities",
}


VALID_RECORD_TYPES = {
    "Items",
    "NPCs",
    "Recipes",
    "Drops",
    "RecipeEntities",
}


def normalize_catalog_name(
    value: str,
) -> str:
    """
    Normalize an entity name for deterministic
    exact matching.

    Examples:
        "Moon Lord"   -> "moonlord"
        "Night’s Edge" -> "nightsedge"
        "Terra Blade" -> "terrablade"
    """
    if not isinstance(value, str):
        raise TypeError(
            "Entity name must be a string."
        )

    text = unicodedata.normalize(
        "NFKC",
        value,
    ).casefold()

    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("`", "'")
    )

    return "".join(
        character
        for character in text
        if character.isalnum()
    )


def resolve_record_type(
    record_type: str,
) -> str:
    """
    Convert a user-facing record type into the
    corresponding SQLite table category.
    """
    if not isinstance(record_type, str):
        raise TypeError(
            "record_type must be a string."
        )

    stripped_value = record_type.strip()

    resolved = RECORD_TYPE_ALIASES.get(
        stripped_value
    )

    if resolved is None:
        resolved = RECORD_TYPE_ALIASES.get(
            stripped_value.casefold()
        )

    if resolved is None:
        raise ValueError(
            f"Unsupported record type: "
            f"{record_type!r}. "
            f"Supported types are: "
            f"{sorted(VALID_RECORD_TYPES)}"
        )

    return resolved


def build_fts_query(
    query: str,
) -> str:
    """
    Convert ordinary user text into a safe FTS5 query.

    Each token is required with AND, so:

        Terra Blade
        -> "Terra" AND "Blade"
    """
    if not isinstance(query, str):
        raise TypeError(
            "Search query must be a string."
        )

    query = query.strip()

    if not query:
        raise ValueError(
            "Search query cannot be empty."
        )

    tokens = re.findall(
        r"[^\W_]+",
        query,
        flags=re.UNICODE,
    )

    if not tokens:
        raise ValueError(
            "Search query contains no searchable text."
        )

    # For normal words, remove isolated punctuation-created
    # letters such as the “s” in "Night's Edge".
    longer_tokens = [
        token
        for token in tokens
        if len(token) > 1
    ]

    if longer_tokens:
        tokens = longer_tokens

    escaped_tokens = [
        token.replace('"', '""')
        for token in tokens
    ]

    return " AND ".join(
        f'"{token}"'
        for token in escaped_tokens
    )


class TerrariaCatalogStore:
    """
    Read-only access layer for the complete Terraria
    Cargo catalog.

    Supports:
    - exact normalized-name lookup;
    - grouped recipe lookup;
    - FTS5 full-text search;
    - lookup by catalog ID;
    - record counts.
    """

    def __init__(
        self,
        database_path: str | Path = (
            DEFAULT_CATALOG_PATH
        ),
    ) -> None:
        self.database_path = Path(
            database_path
        )

        if not self.database_path.exists():
            raise FileNotFoundError(
                "Terraria catalog database "
                f"not found: {self.database_path}"
            )

        self.connection = sqlite3.connect(
            self.database_path
        )

        self.connection.row_factory = (
            sqlite3.Row
        )

        # Prevent accidental modifications.
        self.connection.execute(
            "PRAGMA query_only = ON"
        )

    @classmethod
    def from_default(
        cls,
    ) -> "TerrariaCatalogStore":
        return cls(DEFAULT_CATALOG_PATH)

    def close(self) -> None:
        self.connection.close()

    def __enter__(
        self,
    ) -> "TerrariaCatalogStore":
        return self

    def __exit__(
        self,
        exception_type: Any,
        exception_value: Any,
        traceback: Any,
    ) -> None:
        self.close()

    def counts(self) -> dict[str, int]:
        """
        Return the number of records in each catalog
        category.
        """
        rows = self.connection.execute(
            """
            SELECT
                table_name,
                COUNT(*) AS record_count
            FROM catalog_records
            GROUP BY table_name
            ORDER BY table_name
            """
        ).fetchall()

        counts = {
            row["table_name"]: row["record_count"]
            for row in rows
        }

        grouped_recipe_count = (
            self.connection.execute(
                """
                SELECT COUNT(*) AS record_count
                FROM recipe_entities
                """
            ).fetchone()["record_count"]
        )

        counts["RecipeEntities"] = (
            grouped_recipe_count
        )

        return counts

    def lookup_exact(
        self,
        entity_name: str,
        record_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Find records using exact normalized-name
        matching.

        Returns one of:
            status = found
            status = ambiguous
            status = not_found
        """
        normalized_name = normalize_catalog_name(
            entity_name
        )

        if not normalized_name:
            raise ValueError(
                "Entity name contains no "
                "searchable characters."
            )

        if record_type is None:
            resolved_type = None
        else:
            resolved_type = resolve_record_type(
                record_type
            )

        matches: list[dict[str, Any]] = []

        if (
            resolved_type is None
            or resolved_type == "RecipeEntities"
        ):
            matches.extend(
                self._lookup_recipe_entities(
                    normalized_name
                )
            )

        if (
            resolved_type is None
            or resolved_type != "RecipeEntities"
        ):
            matches.extend(
                self._lookup_catalog_records(
                    normalized_name=normalized_name,
                    record_type=resolved_type,
                )
            )

        if not matches:
            return {
                "status": "not_found",
                "query": entity_name,
                "normalized_query": normalized_name,
                "record_type": resolved_type,
                "matches": [],
            }

        if len(matches) == 1:
            return {
                "status": "found",
                "query": entity_name,
                "normalized_query": normalized_name,
                "record_type": resolved_type,
                "record": matches[0],
            }

        return {
            "status": "ambiguous",
            "query": entity_name,
            "normalized_query": normalized_name,
            "record_type": resolved_type,
            "matches": matches,
        }

    def lookup_recipe(
        self,
        entity_name: str,
    ) -> dict[str, Any]:
        """
        Convenience wrapper for grouped recipes.
        """
        return self.lookup_exact(
            entity_name=entity_name,
            record_type="recipe",
        )

    def get_by_catalog_id(
        self,
        catalog_id: str,
        record_type: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Retrieve one record using its stable catalog ID.
        """
        if not isinstance(catalog_id, str):
            raise TypeError(
                "catalog_id must be a string."
            )

        catalog_id = catalog_id.strip()

        if not catalog_id:
            raise ValueError(
                "catalog_id cannot be empty."
            )

        resolved_type = (
            resolve_record_type(record_type)
            if record_type is not None
            else None
        )

        if (
            resolved_type is None
            or resolved_type == "RecipeEntities"
        ):
            row = self.connection.execute(
                """
                SELECT
                    catalog_id,
                    entity_name,
                    normalized_name,
                    entity_id,
                    payload_json
                FROM recipe_entities
                WHERE catalog_id = ?
                LIMIT 1
                """,
                (catalog_id,),
            ).fetchone()

            if row is not None:
                return self._recipe_row_to_record(
                    row
                )

        if resolved_type == "RecipeEntities":
            return None

        sql = """
            SELECT
                catalog_id,
                table_name,
                entity_name,
                normalized_name,
                entity_id,
                payload_json
            FROM catalog_records
            WHERE catalog_id = ?
        """

        parameters: list[Any] = [
            catalog_id
        ]

        if resolved_type is not None:
            sql += " AND table_name = ?"
            parameters.append(resolved_type)

        sql += " LIMIT 1"

        row = self.connection.execute(
            sql,
            parameters,
        ).fetchone()

        if row is None:
            return None

        return self._catalog_row_to_record(
            row
        )

    def search(
        self,
        query: str,
        record_types: (
            str
            | Iterable[str]
            | None
        ) = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Search entity names and raw payloads using
        SQLite FTS5.

        FTS is used as a fallback when exact lookup
        returns not_found.
        """
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
        ):
            raise TypeError(
                "limit must be an integer."
            )

        if limit <= 0:
            raise ValueError(
                "limit must be greater than zero."
            )

        fts_query = build_fts_query(
            query
        )

        resolved_types = self._resolve_type_list(
            record_types
        )

        sql = """
            SELECT
                catalog_id,
                table_name,
                entity_name,
                bm25(catalog_fts) AS rank
            FROM catalog_fts
            WHERE catalog_fts MATCH ?
        """

        parameters: list[Any] = [
            fts_query
        ]

        if resolved_types is not None:
            placeholders = ", ".join(
                "?"
                for _ in resolved_types
            )

            sql += (
                f" AND table_name IN "
                f"({placeholders})"
            )

            parameters.extend(
                resolved_types
            )

        sql += """
            ORDER BY rank
            LIMIT ?
        """

        parameters.append(limit)

        rows = self.connection.execute(
            sql,
            parameters,
        ).fetchall()

        results = []

        for row in rows:
            record = self.get_by_catalog_id(
                catalog_id=row["catalog_id"],
                record_type=row["table_name"],
            )

            if record is None:
                continue

            results.append(
                {
                    "rank": row["rank"],
                    "record": record,
                }
            )

        return results

    def _lookup_recipe_entities(
        self,
        normalized_name: str,
    ) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT
                catalog_id,
                entity_name,
                normalized_name,
                entity_id,
                payload_json
            FROM recipe_entities
            WHERE normalized_name = ?
            ORDER BY entity_name
            """,
            (normalized_name,),
        ).fetchall()

        return [
            self._recipe_row_to_record(row)
            for row in rows
        ]

    def _lookup_catalog_records(
        self,
        normalized_name: str,
        record_type: str | None,
    ) -> list[dict[str, Any]]:
        sql = """
            SELECT
                catalog_id,
                table_name,
                entity_name,
                normalized_name,
                entity_id,
                payload_json
            FROM catalog_records
            WHERE normalized_name = ?
        """

        parameters: list[Any] = [
            normalized_name
        ]

        if record_type is not None:
            sql += " AND table_name = ?"
            parameters.append(record_type)

        sql += """
            ORDER BY
                table_name,
                entity_name,
                catalog_id
        """

        rows = self.connection.execute(
            sql,
            parameters,
        ).fetchall()

        return [
            self._catalog_row_to_record(row)
            for row in rows
        ]

    def _resolve_type_list(
        self,
        record_types: (
            str
            | Iterable[str]
            | None
        ),
    ) -> list[str] | None:
        if record_types is None:
            return None

        if isinstance(record_types, str):
            values = [record_types]
        else:
            values = list(record_types)

        if not values:
            raise ValueError(
                "record_types cannot be empty."
            )

        resolved = []

        for value in values:
            record_type = resolve_record_type(
                value
            )

            if record_type not in resolved:
                resolved.append(record_type)

        return resolved

    @staticmethod
    def _catalog_row_to_record(
        row: sqlite3.Row,
    ) -> dict[str, Any]:
        return {
            "catalog_id": row["catalog_id"],
            "record_type": row["table_name"],
            "entity_name": row["entity_name"],
            "normalized_name": (
                row["normalized_name"]
            ),
            "entity_id": row["entity_id"],
            "data": json.loads(
                row["payload_json"]
            ),
        }

    @staticmethod
    def _recipe_row_to_record(
        row: sqlite3.Row,
    ) -> dict[str, Any]:
        payload = json.loads(
            row["payload_json"]
        )

        return {
            "catalog_id": row["catalog_id"],
            "record_type": "RecipeEntities",
            "entity_name": row["entity_name"],
            "normalized_name": (
                row["normalized_name"]
            ),
            "entity_id": row["entity_id"],
            "data": payload,
        }
