
from __future__ import annotations

import json
import sqlite3

from pathlib import Path
from typing import Any, Iterable

from .catalog_store import normalize_catalog_name


DEFAULT_DATABASE_PATH = Path(
    "/content/llm_project/data/terraria/"
    "catalog/terraria_query.sqlite3"
)


class TerrariaQueryStore:
    """
    Read-only query interface for the normalized
    Terraria SQLite catalog.

    The store exposes domain-level operations instead
    of requiring callers to write SQL directly.
    """

    def __init__(
        self,
        database_path: str | Path = (
            DEFAULT_DATABASE_PATH
        ),
        *,
        read_only: bool = True,
    ) -> None:
        self.database_path = Path(
            database_path
        )

        if not self.database_path.exists():
            raise FileNotFoundError(
                "Terraria query database not found: "
                f"{self.database_path}"
            )

        self.read_only = read_only

        if read_only:
            database_uri = (
                self.database_path
                .resolve()
                .as_uri()
                + "?mode=ro"
            )

            self._connection = sqlite3.connect(
                database_uri,
                uri=True,
            )

        else:
            self._connection = sqlite3.connect(
                self.database_path
            )

        self._connection.row_factory = (
            sqlite3.Row
        )

        self._connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        self._closed = False

    def __enter__(
        self,
    ) -> "TerrariaQueryStore":
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()

    def _ensure_open(
        self,
    ) -> None:
        if self._closed:
            raise RuntimeError(
                "TerrariaQueryStore is closed."
            )

    def close(
        self,
    ) -> None:
        if not self._closed:
            self._connection.close()
            self._closed = True

    @staticmethod
    def _decode_json(
        value: Any,
        *,
        default: Any = None,
    ) -> Any:
        if value is None:
            return default

        if isinstance(
            value,
            (dict, list),
        ):
            return value

        try:
            return json.loads(
                str(value)
            )

        except (
            json.JSONDecodeError,
            TypeError,
        ):
            return default

    @staticmethod
    def _row_to_dict(
        row: sqlite3.Row | None,
    ) -> dict[str, Any] | None:
        if row is None:
            return None

        return dict(row)

    @staticmethod
    def _rows_to_dicts(
        rows: Iterable[sqlite3.Row],
    ) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in rows
        ]

    def _fetch_one(
        self,
        query: str,
        parameters: tuple[Any, ...] = (),
    ) -> dict[str, Any] | None:
        self._ensure_open()

        row = self._connection.execute(
            query,
            parameters,
        ).fetchone()

        return self._row_to_dict(row)

    def _fetch_all(
        self,
        query: str,
        parameters: tuple[Any, ...] = (),
    ) -> list[dict[str, Any]]:
        self._ensure_open()

        rows = self._connection.execute(
            query,
            parameters,
        ).fetchall()

        return self._rows_to_dicts(rows)

    @staticmethod
    def _validate_limit(
        limit: int,
    ) -> int:
        if isinstance(limit, bool):
            raise TypeError(
                "limit must be an integer."
            )

        if not isinstance(limit, int):
            raise TypeError(
                "limit must be an integer."
            )

        if limit < 1:
            raise ValueError(
                "limit must be at least 1."
            )

        return min(limit, 100)

    @staticmethod
    def _normalize_mode(
        mode: str,
    ) -> str:
        normalized_mode = str(
            mode
        ).strip().casefold()

        valid_modes = {
            "normal",
            "expert",
            "master",
        }

        if normalized_mode not in valid_modes:
            raise ValueError(
                "mode must be one of: "
                "normal, expert, master."
            )

        return normalized_mode

    @staticmethod
    def _fts_query(
        query: str,
    ) -> str:
        tokens = [
            token.strip()
            for token in str(
                query
            ).split()
            if token.strip()
        ]

        if not tokens:
            raise ValueError(
                "Search query cannot be empty."
            )

        escaped_tokens = [
            '"'
            + token.replace(
                '"',
                '""',
            )
            + '"'
            for token in tokens
        ]

        return " AND ".join(
            escaped_tokens
        )

    def metadata(
        self,
    ) -> dict[str, Any]:
        rows = self._fetch_all(
            """
            SELECT
                key,
                value
            FROM metadata
            ORDER BY key
            """
        )

        result: dict[str, Any] = {}

        for row in rows:
            key = row["key"]
            value = row["value"]

            decoded = self._decode_json(
                value,
                default=value,
            )

            result[key] = decoded

        return result

    def counts(
        self,
    ) -> dict[str, int]:
        tables = (
            "items",
            "npcs",
            "recipes",
            "recipe_variants",
            "recipe_stations",
            "recipe_ingredients",
            "drops",
        )

        result: dict[str, int] = {}

        for table in tables:
            row = self._fetch_one(
                f"""
                SELECT COUNT(*) AS count
                FROM {table}
                """
            )

            if row is None:
                raise RuntimeError(
                    f"Unable to count table: "
                    f"{table}"
                )

            result[table] = int(
                row["count"]
            )

        result["total"] = sum(
            result.values()
        )

        return result

    # =================================================
    # Items
    # =================================================

    def get_item(
        self,
        name: str,
        *,
        include_record: bool = True,
    ) -> dict[str, Any]:
        normalized_name = (
            normalize_catalog_name(
                name
            )
        )

        rows = self._fetch_all(
            """
            SELECT
                source_catalog_id,
                item_id,
                name,
                normalized_name,
                internal_name,
                parse_status,
                record_json
            FROM items
            WHERE normalized_name = ?
            ORDER BY
                item_id IS NULL,
                item_id,
                source_catalog_id
            """,
            (
                normalized_name,
            ),
        )

        if not rows:
            return {
                "status": "not_found",
                "query": name,
                "normalized_query": (
                    normalized_name
                ),
                "matches": [],
            }

        parsed_rows = []

        for row in rows:
            if include_record:
                row["record"] = (
                    self._decode_json(
                        row.pop(
                            "record_json"
                        ),
                        default={},
                    )
                )
            else:
                row.pop(
                    "record_json",
                    None,
                )

            parsed_rows.append(row)

        return {
            "status": (
                "found"
                if len(parsed_rows) == 1
                else "ambiguous"
            ),
            "query": name,
            "normalized_query": (
                normalized_name
            ),
            "match": (
                parsed_rows[0]
                if len(parsed_rows) == 1
                else None
            ),
            "matches": parsed_rows,
        }

    def get_item_by_id(
        self,
        item_id: int,
        *,
        include_record: bool = True,
    ) -> dict[str, Any]:
        if isinstance(item_id, bool):
            raise TypeError(
                "item_id must be an integer."
            )

        if not isinstance(item_id, int):
            raise TypeError(
                "item_id must be an integer."
            )

        rows = self._fetch_all(
            """
            SELECT
                source_catalog_id,
                item_id,
                name,
                normalized_name,
                internal_name,
                parse_status,
                record_json
            FROM items
            WHERE item_id = ?
            ORDER BY source_catalog_id
            """,
            (
                item_id,
            ),
        )

        if not rows:
            return {
                "status": "not_found",
                "item_id": item_id,
                "matches": [],
            }

        for row in rows:
            if include_record:
                row["record"] = (
                    self._decode_json(
                        row.pop(
                            "record_json"
                        ),
                        default={},
                    )
                )
            else:
                row.pop(
                    "record_json",
                    None,
                )

        return {
            "status": (
                "found"
                if len(rows) == 1
                else "ambiguous"
            ),
            "item_id": item_id,
            "match": (
                rows[0]
                if len(rows) == 1
                else None
            ),
            "matches": rows,
        }

    def search_items(
        self,
        query: str,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        limit = self._validate_limit(
            limit
        )

        normalized_query = (
            normalize_catalog_name(
                query
            )
        )

        exact_rows = self._fetch_all(
            """
            SELECT
                source_catalog_id,
                item_id,
                name,
                normalized_name,
                internal_name,
                0.0 AS rank,
                'exact' AS match_method
            FROM items
            WHERE normalized_name = ?
            ORDER BY
                item_id IS NULL,
                item_id
            LIMIT ?
            """,
            (
                normalized_query,
                limit,
            ),
        )

        if len(exact_rows) >= limit:
            return exact_rows

        seen_ids = {
            row["source_catalog_id"]
            for row in exact_rows
        }

        remaining = limit - len(
            exact_rows
        )

        try:
            fts_rows = self._fetch_all(
                """
                SELECT
                    i.source_catalog_id,
                    i.item_id,
                    i.name,
                    i.normalized_name,
                    i.internal_name,
                    bm25(items_fts) AS rank,
                    'fts' AS match_method
                FROM items_fts

                JOIN items AS i
                  ON i.source_catalog_id
                     = items_fts.source_catalog_id

                WHERE items_fts MATCH ?

                ORDER BY
                    rank,
                    i.name

                LIMIT ?
                """,
                (
                    self._fts_query(
                        query
                    ),
                    remaining + len(
                        seen_ids
                    ),
                ),
            )

        except sqlite3.OperationalError:
            fts_rows = self._fetch_all(
                """
                SELECT
                    source_catalog_id,
                    item_id,
                    name,
                    normalized_name,
                    internal_name,
                    1.0 AS rank,
                    'prefix' AS match_method
                FROM items
                WHERE normalized_name LIKE ?
                ORDER BY
                    normalized_name,
                    item_id
                LIMIT ?
                """,
                (
                    f"%{normalized_query}%",
                    remaining,
                ),
            )

        results = list(exact_rows)

        for row in fts_rows:
            source_catalog_id = row[
                "source_catalog_id"
            ]

            if source_catalog_id in seen_ids:
                continue

            seen_ids.add(
                source_catalog_id
            )

            results.append(row)

            if len(results) >= limit:
                break

        return results

    # =================================================
    # NPCs
    # =================================================

    def get_npc(
        self,
        name: str,
        *,
        npc_id: int | None = None,
        include_record: bool = True,
    ) -> dict[str, Any]:
        normalized_name = (
            normalize_catalog_name(
                name
            )
        )

        if npc_id is None:
            rows = self._fetch_all(
                """
                SELECT
                    source_catalog_id,
                    npc_id,
                    name,
                    normalized_name,
                    parse_status,
                    record_json
                FROM npcs
                WHERE normalized_name = ?
                ORDER BY
                    npc_id,
                    source_catalog_id
                """,
                (
                    normalized_name,
                ),
            )

        else:
            if isinstance(npc_id, bool):
                raise TypeError(
                    "npc_id must be an integer."
                )

            if not isinstance(npc_id, int):
                raise TypeError(
                    "npc_id must be an integer."
                )

            rows = self._fetch_all(
                """
                SELECT
                    source_catalog_id,
                    npc_id,
                    name,
                    normalized_name,
                    parse_status,
                    record_json
                FROM npcs
                WHERE
                    normalized_name = ?
                    AND npc_id = ?
                ORDER BY source_catalog_id
                """,
                (
                    normalized_name,
                    npc_id,
                ),
            )

        if not rows:
            return {
                "status": "not_found",
                "query": name,
                "normalized_query": (
                    normalized_name
                ),
                "npc_id": npc_id,
                "matches": [],
            }

        for row in rows:
            if include_record:
                row["record"] = (
                    self._decode_json(
                        row.pop(
                            "record_json"
                        ),
                        default={},
                    )
                )
            else:
                row.pop(
                    "record_json",
                    None,
                )

        return {
            "status": (
                "found"
                if len(rows) == 1
                else "family"
            ),
            "query": name,
            "normalized_query": (
                normalized_name
            ),
            "npc_id": npc_id,
            "match": (
                rows[0]
                if len(rows) == 1
                else None
            ),
            "matches": rows,
        }

    def search_npcs(
        self,
        query: str,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        limit = self._validate_limit(
            limit
        )

        normalized_query = (
            normalize_catalog_name(
                query
            )
        )

        exact_rows = self._fetch_all(
            """
            SELECT
                source_catalog_id,
                npc_id,
                name,
                normalized_name,
                0.0 AS rank,
                'exact' AS match_method
            FROM npcs
            WHERE normalized_name = ?
            ORDER BY npc_id
            LIMIT ?
            """,
            (
                normalized_query,
                limit,
            ),
        )

        if len(exact_rows) >= limit:
            return exact_rows

        seen_ids = {
            row["source_catalog_id"]
            for row in exact_rows
        }

        remaining = limit - len(
            exact_rows
        )

        try:
            fts_rows = self._fetch_all(
                """
                SELECT
                    n.source_catalog_id,
                    n.npc_id,
                    n.name,
                    n.normalized_name,
                    bm25(npcs_fts) AS rank,
                    'fts' AS match_method
                FROM npcs_fts

                JOIN npcs AS n
                  ON n.source_catalog_id
                     = npcs_fts.source_catalog_id

                WHERE npcs_fts MATCH ?

                ORDER BY
                    rank,
                    n.name,
                    n.npc_id

                LIMIT ?
                """,
                (
                    self._fts_query(
                        query
                    ),
                    remaining + len(
                        seen_ids
                    ),
                ),
            )

        except sqlite3.OperationalError:
            fts_rows = self._fetch_all(
                """
                SELECT
                    source_catalog_id,
                    npc_id,
                    name,
                    normalized_name,
                    1.0 AS rank,
                    'prefix' AS match_method
                FROM npcs
                WHERE normalized_name LIKE ?
                ORDER BY
                    normalized_name,
                    npc_id
                LIMIT ?
                """,
                (
                    f"%{normalized_query}%",
                    remaining,
                ),
            )

        results = list(exact_rows)

        for row in fts_rows:
            source_catalog_id = row[
                "source_catalog_id"
            ]

            if source_catalog_id in seen_ids:
                continue

            seen_ids.add(
                source_catalog_id
            )

            results.append(row)

            if len(results) >= limit:
                break

        return results

    # =================================================
    # Recipes
    # =================================================

    def get_recipe(
        self,
        result_name: str,
        *,
        preferred_only: bool = True,
        include_record: bool = False,
    ) -> dict[str, Any]:
        normalized_name = (
            normalize_catalog_name(
                result_name
            )
        )

        recipe_rows = self._fetch_all(
            """
            SELECT
                source_catalog_id,
                result_name,
                result_normalized_name,
                result_item_catalog_id,
                result_item_id,
                result_link_status,
                result_link_method,
                linking_status,
                parse_status,
                preferred_variant_ids_json,
                record_json
            FROM recipes
            WHERE result_normalized_name = ?
            ORDER BY source_catalog_id
            """,
            (
                normalized_name,
            ),
        )

        if not recipe_rows:
            return {
                "status": "not_found",
                "query": result_name,
                "normalized_query": (
                    normalized_name
                ),
                "recipes": [],
            }

        recipes = []

        for recipe_row in recipe_rows:
            recipe_catalog_id = recipe_row[
                "source_catalog_id"
            ]

            variant_query = """
                SELECT
                    variant_id,
                    position,
                    is_current,
                    is_legacy,
                    is_preferred,
                    version_label,
                    result_quantity,
                    station_names_json,
                    record_json
                FROM recipe_variants
                WHERE recipe_catalog_id = ?
            """

            parameters: list[Any] = [
                recipe_catalog_id,
            ]

            if preferred_only:
                variant_query += (
                    " AND is_preferred = 1"
                )

            variant_query += (
                " ORDER BY position"
            )

            variant_rows = self._fetch_all(
                variant_query,
                tuple(parameters),
            )

            variants = []

            for variant_row in variant_rows:
                variant_id = variant_row[
                    "variant_id"
                ]

                ingredient_rows = (
                    self._fetch_all(
                        """
                        SELECT
                            position,
                            name,
                            normalized_name,
                            quantity,
                            kind,
                            link_status,
                            link_method,
                            item_catalog_id,
                            item_id,
                            group_json,
                            record_json
                        FROM recipe_ingredients
                        WHERE variant_id = ?
                        ORDER BY position
                        """,
                        (
                            variant_id,
                        ),
                    )
                )

                ingredients = []

                for ingredient in (
                    ingredient_rows
                ):
                    ingredient["group"] = (
                        self._decode_json(
                            ingredient.pop(
                                "group_json"
                            ),
                            default=None,
                        )
                    )

                    if include_record:
                        ingredient["record"] = (
                            self._decode_json(
                                ingredient.pop(
                                    "record_json"
                                ),
                                default={},
                            )
                        )
                    else:
                        ingredient.pop(
                            "record_json",
                            None,
                        )

                    ingredients.append(
                        ingredient
                    )

                variant_row[
                    "stations"
                ] = self._decode_json(
                    variant_row.pop(
                        "station_names_json"
                    ),
                    default=[],
                )

                variant_row[
                    "ingredients"
                ] = ingredients

                if include_record:
                    variant_row["record"] = (
                        self._decode_json(
                            variant_row.pop(
                                "record_json"
                            ),
                            default={},
                        )
                    )
                else:
                    variant_row.pop(
                        "record_json",
                        None,
                    )

                variants.append(
                    variant_row
                )

            recipe_row[
                "preferred_variant_ids"
            ] = self._decode_json(
                recipe_row.pop(
                    "preferred_variant_ids_json"
                ),
                default=[],
            )

            recipe_row["variants"] = variants

            if include_record:
                recipe_row["record"] = (
                    self._decode_json(
                        recipe_row.pop(
                            "record_json"
                        ),
                        default={},
                    )
                )
            else:
                recipe_row.pop(
                    "record_json",
                    None,
                )

            recipes.append(
                recipe_row
            )

        return {
            "status": (
                "found"
                if len(recipes) == 1
                else "ambiguous"
            ),
            "query": result_name,
            "normalized_query": (
                normalized_name
            ),
            "recipe": (
                recipes[0]
                if len(recipes) == 1
                else None
            ),
            "recipes": recipes,
        }

    def search_recipes(
        self,
        query: str,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        limit = self._validate_limit(
            limit
        )

        normalized_query = (
            normalize_catalog_name(
                query
            )
        )

        exact_rows = self._fetch_all(
            """
            SELECT
                source_catalog_id,
                result_name,
                result_normalized_name,
                result_item_id,
                linking_status,
                0.0 AS rank,
                'exact' AS match_method
            FROM recipes
            WHERE result_normalized_name = ?
            ORDER BY source_catalog_id
            LIMIT ?
            """,
            (
                normalized_query,
                limit,
            ),
        )

        if len(exact_rows) >= limit:
            return exact_rows

        seen_ids = {
            row["source_catalog_id"]
            for row in exact_rows
        }

        remaining = limit - len(
            exact_rows
        )

        try:
            fts_rows = self._fetch_all(
                """
                SELECT
                    r.source_catalog_id,
                    r.result_name,
                    r.result_normalized_name,
                    r.result_item_id,
                    r.linking_status,
                    bm25(recipes_fts) AS rank,
                    'fts' AS match_method
                FROM recipes_fts

                JOIN recipes AS r
                  ON r.source_catalog_id
                     = recipes_fts.source_catalog_id

                WHERE recipes_fts MATCH ?

                ORDER BY
                    rank,
                    r.result_name

                LIMIT ?
                """,
                (
                    self._fts_query(
                        query
                    ),
                    remaining + len(
                        seen_ids
                    ),
                ),
            )

        except sqlite3.OperationalError:
            fts_rows = self._fetch_all(
                """
                SELECT
                    source_catalog_id,
                    result_name,
                    result_normalized_name,
                    result_item_id,
                    linking_status,
                    1.0 AS rank,
                    'prefix' AS match_method
                FROM recipes
                WHERE result_normalized_name LIKE ?
                ORDER BY result_normalized_name
                LIMIT ?
                """,
                (
                    f"%{normalized_query}%",
                    remaining,
                ),
            )

        results = list(exact_rows)

        for row in fts_rows:
            source_catalog_id = row[
                "source_catalog_id"
            ]

            if source_catalog_id in seen_ids:
                continue

            seen_ids.add(
                source_catalog_id
            )

            results.append(row)

            if len(results) >= limit:
                break

        return results

    def recipes_using_item(
        self,
        item_name: str,
        *,
        preferred_only: bool = True,
    ) -> dict[str, Any]:
        item_result = self.get_item(
            item_name,
            include_record=False,
        )

        if item_result["status"] == "not_found":
            return {
                "status": "item_not_found",
                "query": item_name,
                "item": None,
                "recipes": [],
            }

        if item_result["status"] == "ambiguous":
            return {
                "status": "item_ambiguous",
                "query": item_name,
                "item": None,
                "item_matches": (
                    item_result["matches"]
                ),
                "recipes": [],
            }

        item = item_result["match"]

        preferred_clause = (
            "AND v.is_preferred = 1"
            if preferred_only
            else ""
        )

        rows = self._fetch_all(
            f"""
            SELECT DISTINCT
                r.source_catalog_id,
                r.result_name,
                r.result_normalized_name,
                r.result_item_id,
                r.linking_status
            FROM recipe_ingredients AS i

            JOIN recipe_variants AS v
              ON v.variant_id
                 = i.variant_id

            JOIN recipes AS r
              ON r.source_catalog_id
                 = v.recipe_catalog_id

            WHERE i.item_catalog_id = ?
              {preferred_clause}

            ORDER BY r.result_name
            """,
            (
                item[
                    "source_catalog_id"
                ],
            ),
        )

        return {
            "status": "found",
            "query": item_name,
            "item": item,
            "recipes": rows,
        }

    # =================================================
    # Drops
    # =================================================

    def _format_drop_row(
        self,
        row: dict[str, Any],
        *,
        mode: str,
        include_record: bool,
    ) -> dict[str, Any]:
        chance_by_mode = self._decode_json(
            row.pop(
                "chance_by_mode_json"
            ),
            default={},
        )

        quantity_by_mode = self._decode_json(
            row.pop(
                "quantity_by_mode_json"
            ),
            default={},
        )

        row["chance_by_mode"] = (
            chance_by_mode
        )

        row["quantity_by_mode"] = (
            quantity_by_mode
        )

        row["quantity_by_condition"] = (
            self._decode_json(
                row.pop(
                    "quantity_by_condition_json"
                ),
                default={},
            )
        )

        row["availability"] = {
            "normal": bool(
                row["available_normal"]
            ),
            "expert": bool(
                row["available_expert"]
            ),
            "master": bool(
                row["available_master"]
            ),
        }

        row["chance"] = chance_by_mode.get(
            mode
        )

        row["quantity_for_mode"] = (
            quantity_by_mode.get(mode)
        )

        row["conditions"] = self._decode_json(
            row.pop(
                "conditions_json"
            ),
            default=[],
        )

        row["item_group"] = self._decode_json(
            row.pop(
                "item_group_json"
            ),
            default=None,
        )

        row["source_group"] = (
            self._decode_json(
                row.pop(
                    "source_group_json"
                ),
                default=None,
            )
        )

        if include_record:
            row["record"] = self._decode_json(
                row.pop(
                    "record_json"
                ),
                default={},
            )
        else:
            row.pop(
                "record_json",
                None,
            )

        return row

    def drops_for_item(
        self,
        item_name: str,
        *,
        mode: str = "normal",
        include_partial: bool = True,
        include_record: bool = False,
    ) -> dict[str, Any]:
        mode = self._normalize_mode(
            mode
        )

        item_result = self.get_item(
            item_name,
            include_record=False,
        )

        if item_result["status"] == "not_found":
            normalized_name = (
                normalize_catalog_name(
                    item_name
                )
            )

            rows = self._fetch_all(
                """
                SELECT *
                FROM drops
                WHERE item_normalized_name = ?
                ORDER BY
                    source_name,
                    source_catalog_id
                """,
                (
                    normalized_name,
                ),
            )

            if not rows:
                return {
                    "status": "item_not_found",
                    "query": item_name,
                    "mode": mode,
                    "item": None,
                    "drops": [],
                }

            resolved_item = None

        elif item_result["status"] == "ambiguous":
            return {
                "status": "item_ambiguous",
                "query": item_name,
                "mode": mode,
                "item": None,
                "item_matches": (
                    item_result["matches"]
                ),
                "drops": [],
            }

        else:
            resolved_item = item_result[
                "match"
            ]

            rows = self._fetch_all(
                """
                SELECT *
                FROM drops
                WHERE item_catalog_id = ?
                ORDER BY
                    source_name,
                    source_catalog_id
                """,
                (
                    resolved_item[
                        "source_catalog_id"
                    ],
                ),
            )

        if not include_partial:
            rows = [
                row
                for row in rows
                if row["linking_status"]
                == "complete"
            ]

        availability_column = (
            f"available_{mode}"
        )

        rows = [
            row
            for row in rows
            if row[
                availability_column
            ] == 1
        ]

        formatted_rows = [
            self._format_drop_row(
                row,
                mode=mode,
                include_record=include_record,
            )
            for row in rows
        ]

        return {
            "status": "found",
            "query": item_name,
            "mode": mode,
            "item": resolved_item,
            "drops": formatted_rows,
        }

    def drops_from_source(
        self,
        source_name: str,
        *,
        mode: str = "normal",
        include_partial: bool = True,
        include_record: bool = False,
    ) -> dict[str, Any]:
        mode = self._normalize_mode(
            mode
        )

        normalized_name = (
            normalize_catalog_name(
                source_name
            )
        )

        rows = self._fetch_all(
            """
            SELECT *
            FROM drops
            WHERE source_normalized_name = ?
            ORDER BY
                item_name,
                source_catalog_id
            """,
            (
                normalized_name,
            ),
        )

        if not rows:
            return {
                "status": "not_found",
                "query": source_name,
                "mode": mode,
                "drops": [],
            }

        if not include_partial:
            rows = [
                row
                for row in rows
                if row["linking_status"]
                == "complete"
            ]

        availability_column = (
            f"available_{mode}"
        )

        rows = [
            row
            for row in rows
            if row[
                availability_column
            ] == 1
        ]

        formatted_rows = [
            self._format_drop_row(
                row,
                mode=mode,
                include_record=include_record,
            )
            for row in rows
        ]

        source_kinds = sorted(
            {
                row["source_kind"]
                for row in formatted_rows
            }
        )

        return {
            "status": "found",
            "query": source_name,
            "normalized_query": (
                normalized_name
            ),
            "mode": mode,
            "source_kinds": source_kinds,
            "drops": formatted_rows,
        }

    # =================================================
    # Unified search
    # =================================================

    def search(
        self,
        query: str,
        *,
        limit_per_type: int = 10,
    ) -> dict[str, Any]:
        limit_per_type = (
            self._validate_limit(
                limit_per_type
            )
        )

        return {
            "query": query,
            "items": self.search_items(
                query,
                limit=limit_per_type,
            ),
            "npcs": self.search_npcs(
                query,
                limit=limit_per_type,
            ),
            "recipes": self.search_recipes(
                query,
                limit=limit_per_type,
            ),
        }
