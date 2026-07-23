
from __future__ import annotations

import hashlib
import json
import sqlite3
import zlib

from collections import Counter
from pathlib import Path
from typing import Any

from src.utils.paths import portable_path, TERRARIA_CATALOG_ROOT


CATALOG_ROOT = TERRARIA_CATALOG_ROOT


DEFAULT_ITEMS_PATH = (
    CATALOG_ROOT
    / "cleaned"
    / "Items.jsonl"
)

DEFAULT_NPCS_PATH = (
    CATALOG_ROOT
    / "cleaned"
    / "NPCs.jsonl"
)

DEFAULT_RECIPES_PATH = (
    CATALOG_ROOT
    / "linked"
    / "Recipes.jsonl"
)

DEFAULT_DROPS_PATH = (
    CATALOG_ROOT
    / "linked"
    / "Drops.jsonl"
)

DEFAULT_DATABASE_PATH = (
    CATALOG_ROOT
    / "terraria_query.sqlite3"
)

DEFAULT_REPORT_PATH = (
    CATALOG_ROOT
    / "terraria_query_report.json"
)


SCHEMA_VERSION = "1.1.0"


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


def _json_text(
    value: Any,
) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _json_blob(
    value: Any,
) -> bytes:
    """Compress a full record payload before storing it in SQLite."""

    return zlib.compress(
        _json_text(value).encode("utf-8"),
        level=9,
    )


def _sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def _linked_target(
    link: Any,
) -> dict[str, Any] | None:
    if not isinstance(link, dict):
        return None

    if link.get("status") != "linked":
        return None

    target = link.get("target")

    if not isinstance(target, dict):
        return None

    return target


def _group_data(
    link: Any,
) -> dict[str, Any] | None:
    if not isinstance(link, dict):
        return None

    group = link.get("group")

    if not isinstance(group, dict):
        return None

    return group


def _create_schema(
    connection: sqlite3.Connection,
) -> None:
    connection.executescript(
        """
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;


        CREATE TABLE items (
            source_catalog_id TEXT PRIMARY KEY,
            item_id INTEGER,
            name TEXT NOT NULL,
            normalized_name TEXT NOT NULL,
            internal_name TEXT,
            parse_status TEXT NOT NULL,
            record_json BLOB NOT NULL
        ) WITHOUT ROWID;


        CREATE TABLE npcs (
            source_catalog_id TEXT PRIMARY KEY,
            npc_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            normalized_name TEXT NOT NULL,
            parse_status TEXT NOT NULL,
            record_json BLOB NOT NULL
        ) WITHOUT ROWID;


        CREATE TABLE recipes (
            source_catalog_id TEXT PRIMARY KEY,

            result_name TEXT NOT NULL,
            result_normalized_name TEXT NOT NULL,

            result_item_catalog_id TEXT,
            result_item_id INTEGER,

            result_link_status TEXT NOT NULL,
            result_link_method TEXT NOT NULL,

            linking_status TEXT NOT NULL,
            parse_status TEXT NOT NULL,

            preferred_variant_ids_json TEXT NOT NULL,
            record_json BLOB NOT NULL,

            FOREIGN KEY (
                result_item_catalog_id
            )
            REFERENCES items (
                source_catalog_id
            )
        ) WITHOUT ROWID;


        CREATE TABLE recipe_variants (
            variant_id TEXT PRIMARY KEY,
            recipe_catalog_id TEXT NOT NULL,

            position INTEGER NOT NULL,
            is_current INTEGER NOT NULL,
            is_legacy INTEGER NOT NULL,
            is_preferred INTEGER NOT NULL,

            version_label TEXT,
            result_quantity INTEGER,

            station_names_json TEXT NOT NULL,
            record_json BLOB NOT NULL,

            FOREIGN KEY (
                recipe_catalog_id
            )
            REFERENCES recipes (
                source_catalog_id
            )
            ON DELETE CASCADE
        ) WITHOUT ROWID;


        CREATE TABLE recipe_stations (
            station_key TEXT PRIMARY KEY,
            variant_id TEXT NOT NULL,

            position INTEGER NOT NULL,
            name TEXT NOT NULL,
            normalized_name TEXT NOT NULL,

            FOREIGN KEY (
                variant_id
            )
            REFERENCES recipe_variants (
                variant_id
            )
            ON DELETE CASCADE
        ) WITHOUT ROWID;


        CREATE TABLE recipe_ingredients (
            ingredient_key TEXT PRIMARY KEY,
            variant_id TEXT NOT NULL,

            position INTEGER NOT NULL,
            name TEXT NOT NULL,
            normalized_name TEXT NOT NULL,
            quantity INTEGER NOT NULL,

            kind TEXT NOT NULL,
            link_status TEXT NOT NULL,
            link_method TEXT NOT NULL,

            item_catalog_id TEXT,
            item_id INTEGER,

            group_json TEXT,
            record_json BLOB NOT NULL,

            FOREIGN KEY (
                variant_id
            )
            REFERENCES recipe_variants (
                variant_id
            )
            ON DELETE CASCADE,

            FOREIGN KEY (
                item_catalog_id
            )
            REFERENCES items (
                source_catalog_id
            )
        ) WITHOUT ROWID;


        CREATE TABLE drops (
            source_catalog_id TEXT PRIMARY KEY,

            item_name TEXT NOT NULL,
            item_normalized_name TEXT NOT NULL,
            item_kind TEXT NOT NULL,
            item_link_status TEXT NOT NULL,
            item_link_method TEXT NOT NULL,

            item_catalog_id TEXT,
            item_id INTEGER,
            item_group_json TEXT,

            source_name TEXT NOT NULL,
            source_normalized_name TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_kind TEXT NOT NULL,
            source_link_status TEXT NOT NULL,
            source_link_method TEXT NOT NULL,

            npc_catalog_id TEXT,
            npc_id INTEGER,
            source_group_json TEXT,

            available_normal INTEGER NOT NULL,
            available_expert INTEGER NOT NULL,
            available_master INTEGER NOT NULL,

            quantity_minimum INTEGER NOT NULL,
            quantity_maximum INTEGER NOT NULL,

            quantity_by_mode_json TEXT NOT NULL,
            quantity_by_condition_json TEXT NOT NULL,
            chance_by_mode_json TEXT NOT NULL,
            conditions_json TEXT NOT NULL,

            linking_status TEXT NOT NULL,
            parse_status TEXT NOT NULL,

            record_json BLOB NOT NULL,

            FOREIGN KEY (
                item_catalog_id
            )
            REFERENCES items (
                source_catalog_id
            ),

            FOREIGN KEY (
                npc_catalog_id
            )
            REFERENCES npcs (
                source_catalog_id
            )
        ) WITHOUT ROWID;
        """
    )


def _create_indexes(
    connection: sqlite3.Connection,
) -> None:
    connection.executescript(
        """
        CREATE INDEX idx_items_name
        ON items (
            normalized_name
        );

        CREATE INDEX idx_items_item_id
        ON items (
            item_id
        );


        CREATE INDEX idx_npcs_name
        ON npcs (
            normalized_name
        );

        CREATE INDEX idx_npcs_npc_id
        ON npcs (
            npc_id
        );


        CREATE INDEX idx_recipes_result_name
        ON recipes (
            result_normalized_name
        );

        CREATE INDEX idx_recipes_result_item
        ON recipes (
            result_item_catalog_id
        );

        CREATE INDEX idx_recipes_linking_status
        ON recipes (
            linking_status
        );


        CREATE INDEX idx_recipe_variants_recipe
        ON recipe_variants (
            recipe_catalog_id,
            position
        );

        CREATE INDEX idx_recipe_variants_preferred
        ON recipe_variants (
            recipe_catalog_id,
            is_preferred
        );


        CREATE INDEX idx_recipe_stations_name
        ON recipe_stations (
            normalized_name
        );

        CREATE INDEX idx_recipe_stations_variant
        ON recipe_stations (
            variant_id,
            position
        );


        CREATE INDEX idx_recipe_ingredients_name
        ON recipe_ingredients (
            normalized_name
        );

        CREATE INDEX idx_recipe_ingredients_item
        ON recipe_ingredients (
            item_catalog_id
        );

        CREATE INDEX idx_recipe_ingredients_variant
        ON recipe_ingredients (
            variant_id,
            position
        );

        CREATE INDEX idx_recipe_ingredients_status
        ON recipe_ingredients (
            link_status
        );


        CREATE INDEX idx_drops_item_name
        ON drops (
            item_normalized_name
        );

        CREATE INDEX idx_drops_item_catalog
        ON drops (
            item_catalog_id
        );

        CREATE INDEX idx_drops_source_name
        ON drops (
            source_normalized_name
        );

        CREATE INDEX idx_drops_npc_catalog
        ON drops (
            npc_catalog_id
        );

        CREATE INDEX idx_drops_source_type
        ON drops (
            source_type
        );

        CREATE INDEX idx_drops_linking_status
        ON drops (
            linking_status
        );
        """
    )


def _create_views(
    connection: sqlite3.Connection,
) -> None:
    connection.executescript(
        """
        CREATE VIEW preferred_recipe_variants AS
        SELECT
            r.source_catalog_id
                AS recipe_catalog_id,

            r.result_name,
            r.result_normalized_name,
            r.result_item_catalog_id,
            r.result_item_id,

            v.variant_id,
            v.position,
            v.is_current,
            v.is_legacy,
            v.version_label,
            v.result_quantity,
            v.station_names_json

        FROM recipes AS r

        JOIN recipe_variants AS v
          ON v.recipe_catalog_id
             = r.source_catalog_id

        WHERE v.is_preferred = 1;


        CREATE VIEW preferred_recipe_ingredients AS
        SELECT
            r.source_catalog_id
                AS recipe_catalog_id,

            r.result_name,
            r.result_normalized_name,

            v.variant_id,
            v.position
                AS variant_position,

            i.position
                AS ingredient_position,

            i.name
                AS ingredient_name,

            i.normalized_name
                AS ingredient_normalized_name,

            i.quantity,
            i.kind,
            i.link_status,
            i.item_catalog_id,
            i.item_id,
            i.group_json

        FROM recipes AS r

        JOIN recipe_variants AS v
          ON v.recipe_catalog_id
             = r.source_catalog_id

        JOIN recipe_ingredients AS i
          ON i.variant_id
             = v.variant_id

        WHERE v.is_preferred = 1;


        CREATE VIEW linked_item_drops AS
        SELECT
            d.source_catalog_id
                AS drop_catalog_id,

            d.item_catalog_id,
            d.item_id,
            d.item_name,

            d.source_name,
            d.source_type,
            d.source_kind,

            d.npc_catalog_id,
            d.npc_id,

            d.available_normal,
            d.available_expert,
            d.available_master,

            d.quantity_minimum,
            d.quantity_maximum,

            d.chance_by_mode_json,
            d.conditions_json,

            d.linking_status,
            d.parse_status

        FROM drops AS d

        WHERE d.item_link_status = 'linked';


        CREATE VIEW linked_npc_drops AS
        SELECT
            d.source_catalog_id
                AS drop_catalog_id,

            d.npc_catalog_id,
            d.npc_id,
            d.source_name,

            d.item_catalog_id,
            d.item_id,
            d.item_name,

            d.available_normal,
            d.available_expert,
            d.available_master,

            d.quantity_minimum,
            d.quantity_maximum,

            d.chance_by_mode_json,
            d.conditions_json,

            d.linking_status,
            d.parse_status

        FROM drops AS d

        WHERE d.source_link_status = 'linked';
        """
    )


def _create_fts(
    connection: sqlite3.Connection,
) -> bool:
    try:
        connection.executescript(
            """
            CREATE VIRTUAL TABLE items_fts
            USING fts5(
                name,
                normalized_name,
                internal_name,
                source_catalog_id UNINDEXED,
                tokenize='unicode61 remove_diacritics 2'
            );


            CREATE VIRTUAL TABLE npcs_fts
            USING fts5(
                name,
                normalized_name,
                source_catalog_id UNINDEXED,
                tokenize='unicode61 remove_diacritics 2'
            );


            CREATE VIRTUAL TABLE recipes_fts
            USING fts5(
                result_name,
                result_normalized_name,
                source_catalog_id UNINDEXED,
                tokenize='unicode61 remove_diacritics 2'
            );


            CREATE VIRTUAL TABLE drops_fts
            USING fts5(
                item_name,
                source_name,
                source_catalog_id UNINDEXED,
                tokenize='unicode61 remove_diacritics 2'
            );
            """
        )

    except sqlite3.OperationalError:
        return False

    connection.execute(
        """
        INSERT INTO items_fts (
            name,
            normalized_name,
            internal_name,
            source_catalog_id
        )
        SELECT
            name,
            normalized_name,
            COALESCE(
                internal_name,
                ''
            ),
            source_catalog_id
        FROM items
        """
    )

    connection.execute(
        """
        INSERT INTO npcs_fts (
            name,
            normalized_name,
            source_catalog_id
        )
        SELECT
            name,
            normalized_name,
            source_catalog_id
        FROM npcs
        """
    )

    connection.execute(
        """
        INSERT INTO recipes_fts (
            result_name,
            result_normalized_name,
            source_catalog_id
        )
        SELECT
            result_name,
            result_normalized_name,
            source_catalog_id
        FROM recipes
        """
    )

    connection.execute(
        """
        INSERT INTO drops_fts (
            item_name,
            source_name,
            source_catalog_id
        )
        SELECT
            item_name,
            source_name,
            source_catalog_id
        FROM drops
        """
    )

    return True


def _insert_metadata(
    connection: sqlite3.Connection,
    metadata: dict[str, Any],
) -> None:
    connection.executemany(
        """
        INSERT INTO metadata (
            key,
            value
        )
        VALUES (?, ?)
        """,
        [
            (
                str(key),
                (
                    value
                    if isinstance(value, str)
                    else _json_text(value)
                ),
            )
            for key, value
            in metadata.items()
        ],
    )


def _insert_items(
    connection: sqlite3.Connection,
    items: list[dict[str, Any]],
) -> None:
    rows = []

    for item in items:
        rows.append(
            (
                item["source_catalog_id"],
                item.get("item_id"),
                item["name"],
                item["normalized_name"],
                item.get("internal_name"),
                item["parse_status"],
                _json_blob(item),
            )
        )

    connection.executemany(
        """
        INSERT INTO items (
            source_catalog_id,
            item_id,
            name,
            normalized_name,
            internal_name,
            parse_status,
            record_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _insert_npcs(
    connection: sqlite3.Connection,
    npcs: list[dict[str, Any]],
) -> None:
    rows = []

    for npc in npcs:
        rows.append(
            (
                npc["source_catalog_id"],
                npc["npc_id"],
                npc["name"],
                npc["normalized_name"],
                npc["parse_status"],
                _json_blob(npc),
            )
        )

    connection.executemany(
        """
        INSERT INTO npcs (
            source_catalog_id,
            npc_id,
            name,
            normalized_name,
            parse_status,
            record_json
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _insert_recipes(
    connection: sqlite3.Connection,
    recipes: list[dict[str, Any]],
) -> dict[str, int]:
    recipe_rows = []
    variant_rows = []
    station_rows = []
    ingredient_rows = []

    for recipe in recipes:
        recipe_catalog_id = recipe[
            "source_catalog_id"
        ]

        result = recipe["result"]

        result_link = result[
            "item_catalog_link"
        ]

        result_target = _linked_target(
            result_link
        )

        result_item_catalog_id = (
            result_target.get(
                "source_catalog_id"
            )
            if result_target
            else None
        )

        result_item_id = (
            result_target.get("item_id")
            if result_target
            else None
        )

        preferred_ids = set(
            recipe[
                "variant_selection"
            ][
                "preferred_variant_ids"
            ]
        )

        recipe_rows.append(
            (
                recipe_catalog_id,

                result["name"],
                result["normalized_name"],

                result_item_catalog_id,
                result_item_id,

                result_link["status"],
                result_link["method"],

                recipe["linking"]["status"],
                recipe["parse_status"],

                _json_text(
                    sorted(
                        preferred_ids
                    )
                ),

                _json_blob(recipe),
            )
        )

        for variant in recipe["variants"]:
            variant_id = variant[
                "variant_id"
            ]

            stations = variant.get(
                "crafting_stations",
                [],
            )

            station_names = [
                station.get("name")
                for station in stations
                if isinstance(
                    station,
                    dict,
                )
            ]

            variant_rows.append(
                (
                    variant_id,
                    recipe_catalog_id,

                    variant["position"],
                    int(
                        bool(
                            variant["is_current"]
                        )
                    ),
                    int(
                        bool(
                            variant["is_legacy"]
                        )
                    ),
                    int(
                        variant_id
                        in preferred_ids
                    ),

                    variant.get(
                        "version",
                        {},
                    ).get(
                        "label"
                    ),

                    variant.get(
                        "result_quantity"
                    ),

                    _json_text(
                        station_names
                    ),

                    _json_blob(variant),
                )
            )

            for station_position, station in enumerate(
                stations
            ):
                station_key = (
                    f"{variant_id}:"
                    f"{station_position}"
                )

                station_rows.append(
                    (
                        station_key,
                        variant_id,
                        station_position,
                        station["name"],
                        station[
                            "normalized_name"
                        ],
                    )
                )

            for ingredient in variant[
                "ingredients"
            ]:
                ingredient_item = ingredient[
                    "item"
                ]

                ingredient_link = (
                    ingredient_item["link"]
                )

                ingredient_target = (
                    _linked_target(
                        ingredient_link
                    )
                )

                item_catalog_id = (
                    ingredient_target.get(
                        "source_catalog_id"
                    )
                    if ingredient_target
                    else None
                )

                item_id = (
                    ingredient_target.get(
                        "item_id"
                    )
                    if ingredient_target
                    else None
                )

                group = _group_data(
                    ingredient_link
                )

                ingredient_position = (
                    ingredient["position"]
                )

                ingredient_key = (
                    f"{variant_id}:"
                    f"{ingredient_position}"
                )

                ingredient_rows.append(
                    (
                        ingredient_key,
                        variant_id,

                        ingredient_position,
                        ingredient_item["name"],
                        ingredient_item[
                            "normalized_name"
                        ],
                        ingredient["quantity"],

                        ingredient_item["kind"],
                        ingredient_link["status"],
                        ingredient_link["method"],

                        item_catalog_id,
                        item_id,

                        (
                            _json_text(group)
                            if group is not None
                            else None
                        ),

                        _json_blob(ingredient),
                    )
                )

    connection.executemany(
        """
        INSERT INTO recipes (
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
        )
        VALUES (
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?
        )
        """,
        recipe_rows,
    )

    connection.executemany(
        """
        INSERT INTO recipe_variants (
            variant_id,
            recipe_catalog_id,

            position,
            is_current,
            is_legacy,
            is_preferred,

            version_label,
            result_quantity,

            station_names_json,
            record_json
        )
        VALUES (
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?
        )
        """,
        variant_rows,
    )

    connection.executemany(
        """
        INSERT INTO recipe_stations (
            station_key,
            variant_id,
            position,
            name,
            normalized_name
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        station_rows,
    )

    connection.executemany(
        """
        INSERT INTO recipe_ingredients (
            ingredient_key,
            variant_id,

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
        )
        VALUES (
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?
        )
        """,
        ingredient_rows,
    )

    return {
        "recipes": len(recipe_rows),
        "recipe_variants": len(
            variant_rows
        ),
        "recipe_stations": len(
            station_rows
        ),
        "recipe_ingredients": len(
            ingredient_rows
        ),
    }


def _insert_drops(
    connection: sqlite3.Connection,
    drops: list[dict[str, Any]],
) -> None:
    rows = []

    for drop in drops:
        item = drop["item"]
        source = drop["source"]

        item_link = item[
            "catalog_link"
        ]

        source_link = source[
            "catalog_link"
        ]

        item_target = _linked_target(
            item_link
        )

        source_target = _linked_target(
            source_link
        )

        item_group = _group_data(
            item_link
        )

        source_group = _group_data(
            source_link
        )

        quantity = drop["quantity"]
        quantity_default = quantity[
            "default"
        ]

        chance = drop["chance"]
        availability = drop[
            "availability"
        ]

        rows.append(
            (
                drop["source_catalog_id"],

                item["name"],
                item["normalized_name"],
                item["kind"],
                item_link["status"],
                item_link["method"],

                (
                    item_target.get(
                        "source_catalog_id"
                    )
                    if item_target
                    else None
                ),

                (
                    item_target.get(
                        "item_id"
                    )
                    if item_target
                    else None
                ),

                (
                    _json_text(item_group)
                    if item_group is not None
                    else None
                ),

                source["name"],
                source["normalized_name"],
                source["source_type"],
                source["kind"],
                source_link["status"],
                source_link["method"],

                (
                    source_target.get(
                        "source_catalog_id"
                    )
                    if source_target
                    else None
                ),

                (
                    source_target.get(
                        "npc_id"
                    )
                    if source_target
                    else None
                ),

                (
                    _json_text(
                        source_group
                    )
                    if source_group is not None
                    else None
                ),

                int(
                    bool(
                        availability["normal"]
                    )
                ),

                int(
                    bool(
                        availability["expert"]
                    )
                ),

                int(
                    bool(
                        availability["master"]
                    )
                ),

                quantity_default["minimum"],
                quantity_default["maximum"],

                _json_text(
                    quantity["by_mode"]
                ),

                _json_text(
                    quantity["by_condition"]
                ),

                _json_text(
                    chance["by_mode"]
                ),

                _json_text(
                    drop["conditions"]
                ),

                drop["linking"]["status"],
                drop["parse_status"],

                _json_blob(drop),
            )
        )

    connection.executemany(
        """
        INSERT INTO drops (
            source_catalog_id,

            item_name,
            item_normalized_name,
            item_kind,
            item_link_status,
            item_link_method,

            item_catalog_id,
            item_id,
            item_group_json,

            source_name,
            source_normalized_name,
            source_type,
            source_kind,
            source_link_status,
            source_link_method,

            npc_catalog_id,
            npc_id,
            source_group_json,

            available_normal,
            available_expert,
            available_master,

            quantity_minimum,
            quantity_maximum,

            quantity_by_mode_json,
            quantity_by_condition_json,
            chance_by_mode_json,
            conditions_json,

            linking_status,
            parse_status,

            record_json
        )
        VALUES (
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?
        )
        """,
        rows,
    )


def _table_count(
    connection: sqlite3.Connection,
    table_name: str,
) -> int:
    row = connection.execute(
        f"SELECT COUNT(*) FROM {table_name}"
    ).fetchone()

    if row is None:
        raise RuntimeError(
            f"Could not count table: "
            f"{table_name}"
        )

    return int(row[0])


def build_query_database(
    items_path: str | Path = (
        DEFAULT_ITEMS_PATH
    ),
    npcs_path: str | Path = (
        DEFAULT_NPCS_PATH
    ),
    recipes_path: str | Path = (
        DEFAULT_RECIPES_PATH
    ),
    drops_path: str | Path = (
        DEFAULT_DROPS_PATH
    ),
    database_path: str | Path = (
        DEFAULT_DATABASE_PATH
    ),
    report_path: str | Path = (
        DEFAULT_REPORT_PATH
    ),
) -> dict[str, Any]:
    items_path = Path(items_path)
    npcs_path = Path(npcs_path)
    recipes_path = Path(recipes_path)
    drops_path = Path(drops_path)

    database_path = Path(
        database_path
    )

    report_path = Path(
        report_path
    )

    items = _load_jsonl(
        items_path
    )

    npcs = _load_jsonl(
        npcs_path
    )

    recipes = _load_jsonl(
        recipes_path
    )

    drops = _load_jsonl(
        drops_path
    )

    database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_database_path = (
        database_path.with_suffix(
            database_path.suffix + ".tmp"
        )
    )

    temporary_database_path.unlink(
        missing_ok=True
    )

    connection = sqlite3.connect(
        temporary_database_path
    )

    connection.row_factory = (
        sqlite3.Row
    )

    try:
        connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        connection.execute(
            "PRAGMA journal_mode = OFF"
        )

        connection.execute(
            "PRAGMA synchronous = OFF"
        )

        connection.execute(
            "PRAGMA temp_store = MEMORY"
        )

        _create_schema(
            connection
        )

        connection.execute("BEGIN")

        _insert_metadata(
            connection,
            {
                "schema_version": (
                    SCHEMA_VERSION
                ),

                "database_type": (
                    "terraria_query_catalog"
                ),
                "record_json_encoding": "zlib+utf-8",

                "source_files": {
                    "items": str(items_path),
                    "npcs": str(npcs_path),
                    "recipes": str(
                        recipes_path
                    ),
                    "drops": str(drops_path),
                },

                "source_sha256": {
                    "items": _sha256(
                        items_path
                    ),
                    "npcs": _sha256(
                        npcs_path
                    ),
                    "recipes": _sha256(
                        recipes_path
                    ),
                    "drops": _sha256(
                        drops_path
                    ),
                },
            },
        )

        _insert_items(
            connection,
            items,
        )

        _insert_npcs(
            connection,
            npcs,
        )

        recipe_insert_counts = (
            _insert_recipes(
                connection,
                recipes,
            )
        )

        _insert_drops(
            connection,
            drops,
        )

        _create_indexes(
            connection
        )

        _create_views(
            connection
        )

        fts_enabled = _create_fts(
            connection
        )

        connection.execute("ANALYZE")
        connection.execute("PRAGMA optimize")
        connection.commit()

        foreign_key_errors = (
            connection.execute(
                "PRAGMA foreign_key_check"
            ).fetchall()
        )

        if foreign_key_errors:
            raise AssertionError(
                "SQLite foreign-key check "
                f"failed: {foreign_key_errors[:10]}"
            )

        integrity_result = (
            connection.execute(
                "PRAGMA integrity_check"
            ).fetchone()
        )

        if (
            integrity_result is None
            or integrity_result[0] != "ok"
        ):
            raise AssertionError(
                "SQLite integrity check failed: "
                f"{integrity_result}"
            )

        table_names = [
            "metadata",
            "items",
            "npcs",
            "recipes",
            "recipe_variants",
            "recipe_stations",
            "recipe_ingredients",
            "drops",
        ]

        table_counts = {
            table_name: _table_count(
                connection,
                table_name,
            )
            for table_name in table_names
        }

        status_counts = {
            "recipes": dict(
                connection.execute(
                    """
                    SELECT
                        linking_status,
                        COUNT(*)
                    FROM recipes
                    GROUP BY linking_status
                    ORDER BY linking_status
                    """
                ).fetchall()
            ),

            "recipe_ingredients": dict(
                connection.execute(
                    """
                    SELECT
                        link_status,
                        COUNT(*)
                    FROM recipe_ingredients
                    GROUP BY link_status
                    ORDER BY link_status
                    """
                ).fetchall()
            ),

            "drops": dict(
                connection.execute(
                    """
                    SELECT
                        linking_status,
                        COUNT(*)
                    FROM drops
                    GROUP BY linking_status
                    ORDER BY linking_status
                    """
                ).fetchall()
            ),
        }

        report = {
            "status": "passed",

            "schema_version": (
                SCHEMA_VERSION
            ),

            "database_path": portable_path(database_path),

            "database_size_bytes": (
                temporary_database_path
                .stat()
                .st_size
            ),

            "fts_enabled": fts_enabled,

            "input_counts": {
                "items": len(items),
                "npcs": len(npcs),
                "recipes": len(recipes),
                "drops": len(drops),
            },

            "insert_counts": {
                "items": len(items),
                "npcs": len(npcs),
                **recipe_insert_counts,
                "drops": len(drops),
            },

            "table_counts": table_counts,

            "status_counts": (
                status_counts
            ),

            "integrity": {
                "sqlite_integrity_check": (
                    "ok"
                ),
                "foreign_key_errors": 0,
            },

            "source_sha256": {
                "items": _sha256(
                    items_path
                ),
                "npcs": _sha256(
                    npcs_path
                ),
                "recipes": _sha256(
                    recipes_path
                ),
                "drops": _sha256(
                    drops_path
                ),
            },
        }

    except Exception:
        connection.rollback()
        connection.close()

        temporary_database_path.unlink(
            missing_ok=True
        )

        raise

    else:
        connection.close()

    temporary_database_path.replace(
        database_path
    )

    report[
        "database_size_bytes"
    ] = database_path.stat().st_size

    report[
        "database_sha256"
    ] = _sha256(
        database_path
    )

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
