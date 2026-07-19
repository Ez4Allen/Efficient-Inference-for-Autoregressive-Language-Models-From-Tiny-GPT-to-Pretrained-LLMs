
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sqlite3
import time
import unicodedata

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


PROJECT_ROOT = Path("/content/llm_project")

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "terraria"
)

CATALOG_PATH = (
    DATA_ROOT
    / "catalog"
)

API_URL = (
    "https://terraria.wiki.gg/api.php"
)

CORE_TABLES = (
    "Items",
    "NPCs",
    "Recipes",
    "Drops",
)

PAGE_SIZE = 500

USER_AGENT = (
    "Terraria-LLM-Catalog/1.0 "
    "(https://github.com/Ez4Allen/"
    "Efficient-Inference-for-Autoregressive-"
    "Language-Models-From-Tiny-GPT-to-"
    "Pretrained-LLMs)"
)


def build_session() -> requests.Session:
    """
    Create one HTTP session with retry support.
    """
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        backoff_factor=1.0,
        status_forcelist=[
            429,
            500,
            502,
            503,
            504,
        ],
        allowed_methods=[
            "GET",
            "POST",
        ],
    )

    adapter = HTTPAdapter(
        max_retries=retry,
    )

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        }
    )

    session.mount(
        "https://",
        adapter,
    )

    return session


SESSION = build_session()


def api_request(
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """
    Send one MediaWiki Action API request.

    POST is used because all-field Cargo queries can
    produce long parameter strings.
    """
    payload = {
        "format": "json",
        "formatversion": 2,
        **parameters,
    }

    response = SESSION.post(
        API_URL,
        data=payload,
        timeout=120,
    )

    response.raise_for_status()

    try:
        result = response.json()

    except ValueError as error:
        preview = response.text[:500]

        raise RuntimeError(
            "The Terraria Wiki API did not return JSON.\n"
            f"Response preview:\n{preview}"
        ) from error

    if "error" in result:
        raise RuntimeError(
            "Terraria Wiki API error:\n"
            + json.dumps(
                result["error"],
                ensure_ascii=False,
                indent=2,
            )
        )

    return result


def unique_strings(
    values: list[str],
) -> list[str]:
    """
    Preserve order while removing duplicates.
    """
    output = []
    seen = set()

    for value in values:
        if not isinstance(value, str):
            continue

        value = value.strip()

        if not value:
            continue

        if value in seen:
            continue

        seen.add(value)
        output.append(value)

    return output


def extract_table_names(
    payload: Any,
) -> list[str]:
    """
    Extract table names from different possible
    Cargo API response shapes.
    """
    names: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for element in value:
                visit(element)

            return

        if not isinstance(value, dict):
            return

        candidate_keys = {
            "name",
            "table",
            "tablename",
            "table_name",
            "tableName",
            "_tableName",
        }

        for key, candidate in value.items():
            if (
                key in candidate_keys
                and isinstance(candidate, str)
            ):
                names.append(candidate)

        for nested in value.values():
            if isinstance(nested, (dict, list)):
                visit(nested)

    visit(payload)

    return unique_strings(names)


def get_all_cargo_tables() -> list[str]:
    """
    Discover all Cargo tables available on the wiki.
    """
    payload = api_request(
        {
            "action": "cargotables",
        }
    )

    names = extract_table_names(
        payload.get(
            "cargotables",
            payload,
        )
    )

    # Remove internal/helper tables.
    names = [
        name
        for name in names
        if not name.startswith("_")
        and "__" not in name
    ]

    # Core tables must always be included.
    names = unique_strings(
        [
            *CORE_TABLES,
            *names,
        ]
    )

    return sorted(
        names,
        key=str.casefold,
    )


def extract_field_names(
    payload: Any,
) -> list[str]:
    """
    Extract field names from cargofields output.
    """
    names: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for element in value:
                visit(element)

            return

        if not isinstance(value, dict):
            return

        candidate_keys = {
            "name",
            "field",
            "fieldname",
            "field_name",
            "fieldName",
        }

        for key, candidate in value.items():
            if (
                key in candidate_keys
                and isinstance(candidate, str)
            ):
                names.append(candidate)

        # Some Cargo versions return:
        # {"field_name": {"type": "..."}}
        for key, nested in value.items():
            if (
                isinstance(nested, dict)
                and key not in {
                    "cargofields",
                    "fields",
                    "query",
                }
                and not key.startswith("_")
            ):
                if any(
                    metadata_key in nested
                    for metadata_key in [
                        "type",
                        "fieldType",
                        "isList",
                        "description",
                    ]
                ):
                    names.append(key)

            if isinstance(nested, (dict, list)):
                visit(nested)

    visit(payload)

    return unique_strings(names)


def get_table_fields(
    table_name: str,
) -> list[str]:
    """
    Fetch all declared fields for one Cargo table.

    Different Cargo versions have used slightly
    different parameter names, so three variants
    are attempted.
    """
    attempts = [
        {
            "action": "cargofields",
            "table": table_name,
        },
        {
            "action": "cargofields",
            "table_name": table_name,
        },
        {
            "action": "cargofields",
            "tables": table_name,
        },
    ]

    last_error: Exception | None = None

    for parameters in attempts:
        try:
            payload = api_request(parameters)

            fields = extract_field_names(
                payload.get(
                    "cargofields",
                    payload.get(
                        "fields",
                        payload,
                    ),
                )
            )

            if fields:
                break

        except Exception as error:
            last_error = error

    else:
        if last_error is not None:
            raise RuntimeError(
                f"Unable to discover fields for "
                f"Cargo table {table_name!r}."
            ) from last_error

        raise RuntimeError(
            f"No fields were returned for "
            f"Cargo table {table_name!r}."
        )

    system_fields = [
        "_rowID",
        "_pageName",
        "_pageID",
        "_pageNamespace",
    ]

    return unique_strings(
        [
            *system_fields,
            *fields,
        ]
    )


def unwrap_cargo_rows(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Convert Cargo's wrapper format into plain rows.
    """
    raw_rows = payload.get(
        "cargoquery",
        [],
    )

    rows: list[dict[str, Any]] = []

    for entry in raw_rows:
        if not isinstance(entry, dict):
            continue

        row = entry.get(
            "title",
            entry,
        )

        if isinstance(row, dict):
            rows.append(row)

    return rows


def stable_json(
    value: Any,
) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def row_identity(
    table_name: str,
    row: dict[str, Any],
) -> str:
    """
    Produce a stable row identity for deduplication.
    """
    row_id = case_insensitive_get(
        row,
        "_rowID",
    )

    if row_id not in {
        None,
        "",
    }:
        return (
            f"{table_name}:row:{row_id}"
        )

    digest = hashlib.sha1(
        stable_json(row).encode("utf-8")
    ).hexdigest()

    return (
        f"{table_name}:sha1:{digest}"
    )


def download_table(
    table_name: str,
    fields: list[str],
) -> list[dict[str, Any]]:
    """
    Download one complete Cargo table using
    offset pagination.
    """
    rows: list[dict[str, Any]] = []
    seen_identities: set[str] = set()

    offset = 0

    field_expression = ",".join(fields)

    while True:
        payload = api_request(
            {
                "action": "cargoquery",
                "tables": table_name,
                "fields": field_expression,
                "order_by": "_rowID",
                "limit": PAGE_SIZE,
                "offset": offset,
            }
        )

        page_rows = unwrap_cargo_rows(
            payload
        )

        new_rows = 0

        for row in page_rows:
            identity = row_identity(
                table_name,
                row,
            )

            if identity in seen_identities:
                continue

            seen_identities.add(identity)
            rows.append(row)
            new_rows += 1

        print(
            f"  offset={offset:<7} "
            f"received={len(page_rows):<6} "
            f"new={new_rows:<6} "
            f"total={len(rows)}"
        )

        # Only an empty page proves that the table
        # has been fully consumed. The wiki may enforce
        # a lower server-side limit than requested.
        if not page_rows:
            break

        if new_rows == 0:
            raise RuntimeError(
                f"Pagination for {table_name!r} "
                "stopped making progress."
            )

        # Advance by the number of rows actually returned,
        # rather than assuming the requested limit was used.
        offset += len(page_rows)
        time.sleep(0.25)

    return rows


def case_insensitive_get(
    record: dict[str, Any],
    *candidate_names: str,
) -> Any:
    """
    Read a dictionary key without depending on case.
    """
    key_map = {
        str(key).casefold(): key
        for key in record
    }

    for candidate in candidate_names:
        original_key = key_map.get(
            candidate.casefold()
        )

        if original_key is not None:
            return record[original_key]

    return None


def normalize_name(
    value: Any,
) -> str:
    """
    Normalize names for deterministic exact lookup.
    """
    if value is None:
        return ""

    text = unicodedata.normalize(
        "NFKC",
        str(value),
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


def clean_string(
    value: Any,
) -> str | None:
    if value is None:
        return None

    text = str(value).strip()

    return text or None


def guess_entity_name(
    table_name: str,
    row: dict[str, Any],
) -> str | None:
    """
    Extract the primary human-readable entity name.
    """
    table_key = table_name.casefold()

    table_specific_candidates = {
        "recipes": [
            "result",
            "resultname",
        ],
        "drops": [
            "item",
            "drop",
            "result",
            "source",
        ],
        "items": [
            "name",
            "item",
        ],
        "npcs": [
            "nameraw",
            "name",
            "npc",
        ],
    }

    candidates = [
        *table_specific_candidates.get(
            table_key,
            [],
        ),
        "name",
        "title",
        "entity",
        "item",
        "npc",
        "result",
        "_pageName",
    ]

    value = case_insensitive_get(
        row,
        *candidates,
    )

    return clean_string(value)


def guess_entity_id(
    table_name: str,
    row: dict[str, Any],
) -> str | None:
    table_key = table_name.casefold()

    candidates = {
        "recipes": [
            "resultid",
            "result_id",
        ],
        "items": [
            "id",
            "itemid",
            "item_id",
        ],
        "npcs": [
            "id",
            "npcid",
            "npc_id",
        ],
        "drops": [
            "itemid",
            "item_id",
            "sourceid",
            "source_id",
        ],
    }.get(
        table_key,
        [],
    )

    value = case_insensitive_get(
        row,
        *candidates,
        "id",
        "_rowID",
    )

    return clean_string(value)


def safe_filename(
    table_name: str,
) -> str:
    normalized = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        table_name,
    ).strip("_")

    return normalized or "unnamed_table"


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def make_catalog_record(
    table_name: str,
    row: dict[str, Any],
) -> dict[str, Any]:
    entity_name = guess_entity_name(
        table_name,
        row,
    )

    entity_id = guess_entity_id(
        table_name,
        row,
    )

    return {
        "catalog_id": row_identity(
            table_name,
            row,
        ),
        "record_type": table_name,
        "entity_name": entity_name,
        "normalized_name": normalize_name(
            entity_name
        ),
        "entity_id": entity_id,
        "data": row,
    }


def parse_numeric(
    value: Any,
) -> int | float | None:
    if value is None:
        return None

    text = str(value).strip()

    if re.fullmatch(
        r"[+-]?\d+",
        text,
    ):
        return int(text)

    if re.fullmatch(
        r"[+-]?(?:\d+\.\d*|\.\d+)",
        text,
    ):
        return float(text)

    return None


def clean_item_expression(
    value: str,
) -> str:
    """
    Remove recipe image/note annotations while
    preserving alternative ingredient names.
    """
    value = re.sub(
        r"#i:[^#]+",
        "",
        value,
    )

    value = re.sub(
        r"#n:[^#]+",
        "",
        value,
    )

    value = value.strip()

    if "¦" in value:
        alternatives = [
            part.strip()
            for part in value.split("¦")
            if part.strip()
        ]

        if alternatives:
            return " or ".join(
                unique_strings(alternatives)
            )

    return value


def parse_recipe_ingredients(
    args_value: Any,
) -> list[dict[str, Any]]:
    """
    Parse the Cargo Recipes.args format:

        item¦amount^item¦amount
    """
    text = clean_string(args_value)

    if not text:
        return []

    ingredients = []

    for raw_part in text.split("^"):
        raw_part = raw_part.strip()

        if not raw_part:
            continue

        if "¦" in raw_part:
            item_raw, quantity_raw = (
                raw_part.rsplit(
                    "¦",
                    maxsplit=1,
                )
            )
        else:
            item_raw = raw_part
            quantity_raw = ""

        item_name = clean_item_expression(
            item_raw
        )

        quantity = parse_numeric(
            quantity_raw
        )

        ingredient = {
            "item": item_name,
            "quantity": quantity,
        }

        if quantity is None:
            ingredient["quantity_raw"] = (
                quantity_raw.strip()
            )

        ingredients.append(
            ingredient
        )

    return ingredients


def parse_alternatives(
    value: Any,
) -> list[str]:
    text = clean_string(value)

    if not text:
        return []

    if "¦" in text:
        values = [
            element.strip()
            for element in text.split("¦")
            if element.strip()
        ]

        if values:
            return unique_strings(values)

    if " / " in text:
        values = [
            element.strip()
            for element in text.split(" / ")
            if element.strip()
        ]

        if values:
            return unique_strings(values)

    return [text]


def build_grouped_recipes(
    recipe_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Merge individual Cargo recipe rows into one
    record per crafted result.
    """
    grouped: dict[
        tuple[str, str],
        dict[str, Any],
    ] = {}

    for row in recipe_rows:
        raw_entity_name = clean_string(
            case_insensitive_get(
                row,
                "result",
            )
        )

        entity_name = (
            clean_item_expression(raw_entity_name)
            if raw_entity_name
            else None
        )

        entity_id = clean_string(
            case_insensitive_get(
                row,
                "resultid",
            )
        )

        if not entity_name:
            continue

        key = (
            entity_id or "",
            entity_name,
        )

        record = grouped.setdefault(
            key,
            {
                "catalog_id": (
                    "recipe:"
                    + hashlib.sha1(
                        (
                            (entity_id or "")
                            + "|"
                            + normalize_name(entity_name)
                        ).encode("utf-8")
                    ).hexdigest()
                ),
                "record_type": "recipe",
                "entity_name": entity_name,
                "normalized_name": normalize_name(
                    entity_name
                ),
                "entity_id": entity_id,
                "recipe_variants": [],
            },
        )

        variant_payload = {
            "variant_id": (
                "variant_"
                + hashlib.sha1(
                    stable_json(row).encode(
                        "utf-8"
                    )
                ).hexdigest()[:12]
            ),
            "result_quantity": parse_numeric(
                case_insensitive_get(
                    row,
                    "amount",
                )
            ),
            "result_quantity_raw": clean_string(
                case_insensitive_get(
                    row,
                    "amount",
                )
            ),
            "ingredients": (
                parse_recipe_ingredients(
                    case_insensitive_get(
                        row,
                        "args",
                    )
                )
            ),
            "crafting_stations": (
                parse_alternatives(
                    case_insensitive_get(
                        row,
                        "station",
                    )
                )
            ),
            "version": clean_string(
                case_insensitive_get(
                    row,
                    "version",
                )
            ),
            "legacy": clean_string(
                case_insensitive_get(
                    row,
                    "legacy",
                )
            ),
            "note": clean_string(
                case_insensitive_get(
                    row,
                    "resulttext",
                    "note",
                )
            ),
            "raw": row,
        }

        record["recipe_variants"].append(
            variant_payload
        )

    output = list(grouped.values())

    output.sort(
        key=lambda record: (
            record["entity_name"].casefold()
        )
    )

    return output


def create_database(
    path: Path,
) -> sqlite3.Connection:
    connection = sqlite3.connect(path)

    connection.executescript(
        """
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;

        CREATE TABLE catalog_records (
            catalog_id TEXT PRIMARY KEY,
            table_name TEXT NOT NULL,
            entity_name TEXT,
            normalized_name TEXT,
            entity_id TEXT,
            payload_json TEXT NOT NULL
        );

        CREATE INDEX idx_catalog_table
        ON catalog_records(table_name);

        CREATE INDEX idx_catalog_normalized_name
        ON catalog_records(normalized_name);

        CREATE INDEX idx_catalog_entity_id
        ON catalog_records(entity_id);

        CREATE TABLE recipe_entities (
            catalog_id TEXT PRIMARY KEY,
            entity_name TEXT NOT NULL,
            normalized_name TEXT NOT NULL,
            entity_id TEXT,
            payload_json TEXT NOT NULL
        );

        CREATE INDEX idx_recipe_normalized_name
        ON recipe_entities(normalized_name);

        CREATE VIRTUAL TABLE catalog_fts
        USING fts5(
            catalog_id UNINDEXED,
            table_name UNINDEXED,
            entity_name,
            payload
        );

        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )

    return connection


def insert_catalog_records(
    connection: sqlite3.Connection,
    records: list[dict[str, Any]],
) -> None:
    rows = []

    fts_rows = []

    for record in records:
        payload_json = json.dumps(
            record["data"],
            ensure_ascii=False,
        )

        rows.append(
            (
                record["catalog_id"],
                record["record_type"],
                record["entity_name"],
                record["normalized_name"],
                record["entity_id"],
                payload_json,
            )
        )

        fts_rows.append(
            (
                record["catalog_id"],
                record["record_type"],
                record["entity_name"] or "",
                payload_json,
            )
        )

    connection.executemany(
        """
        INSERT OR REPLACE INTO catalog_records (
            catalog_id,
            table_name,
            entity_name,
            normalized_name,
            entity_id,
            payload_json
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )

    connection.executemany(
        """
        INSERT INTO catalog_fts (
            catalog_id,
            table_name,
            entity_name,
            payload
        )
        VALUES (?, ?, ?, ?)
        """,
        fts_rows,
    )

    connection.commit()


def insert_grouped_recipes(
    connection: sqlite3.Connection,
    recipes: list[dict[str, Any]],
) -> None:
    rows = []

    fts_rows = []

    for recipe in recipes:
        payload_json = json.dumps(
            recipe,
            ensure_ascii=False,
        )

        rows.append(
            (
                recipe["catalog_id"],
                recipe["entity_name"],
                recipe["normalized_name"],
                recipe["entity_id"],
                payload_json,
            )
        )

        fts_rows.append(
            (
                recipe["catalog_id"],
                "RecipeEntities",
                recipe["entity_name"],
                payload_json,
            )
        )

    connection.executemany(
        """
        INSERT OR REPLACE INTO recipe_entities (
            catalog_id,
            entity_name,
            normalized_name,
            entity_id,
            payload_json
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )

    connection.executemany(
        """
        INSERT INTO catalog_fts (
            catalog_id,
            table_name,
            entity_name,
            payload
        )
        VALUES (?, ?, ?, ?)
        """,
        fts_rows,
    )

    connection.commit()


def write_attribution(
    build_path: Path,
) -> None:
    text = """# Terraria Catalog Attribution

Source: Official Terraria Wiki
Source site: https://terraria.wiki.gg/
Import method: MediaWiki Cargo API

The imported wiki data remains subject to the
licensing and attribution requirements stated by
the Official Terraria Wiki.

This catalog was generated automatically. It is not
an official Re-Logic or Terraria product.
"""

    (
        build_path
        / "ATTRIBUTION.md"
    ).write_text(
        text,
        encoding="utf-8",
    )


def create_build_directory() -> Path:
    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    build_path = (
        DATA_ROOT
        / f"catalog_build_{timestamp}"
    )

    build_path.mkdir(
        parents=True,
        exist_ok=False,
    )

    return build_path


def finalize_catalog(
    build_path: Path,
    replace: bool,
) -> Path | None:
    """
    Atomically replace the active catalog directory.
    """
    backup_path = None

    if CATALOG_PATH.exists():
        if not replace:
            raise FileExistsError(
                f"{CATALOG_PATH} already exists. "
                "Use --replace to replace it."
            )

        timestamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%dT%H%M%SZ")

        backup_path = (
            DATA_ROOT
            / f"catalog_backup_{timestamp}"
        )

        CATALOG_PATH.replace(
            backup_path
        )

    try:
        build_path.replace(
            CATALOG_PATH
        )

    except Exception:
        if (
            backup_path is not None
            and backup_path.exists()
            and not CATALOG_PATH.exists()
        ):
            backup_path.replace(
                CATALOG_PATH
            )

        raise

    return backup_path


def run_import(
    all_tables: bool,
    replace: bool,
) -> None:
    DATA_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    build_path = create_build_directory()

    raw_directory = (
        build_path / "raw"
    )

    normalized_directory = (
        build_path / "normalized"
    )

    raw_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    normalized_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    database_path = (
        build_path
        / "terraria_catalog.sqlite3"
    )

    connection = create_database(
        database_path
    )

    imported_at = datetime.now(
        timezone.utc
    ).isoformat()

    table_counts: dict[str, int] = {}
    failures: dict[str, str] = {}
    recipe_rows: list[dict[str, Any]] = []

    try:
        if all_tables:
            print("Discovering Cargo tables...")

            table_names = (
                get_all_cargo_tables()
            )

        else:
            table_names = list(CORE_TABLES)

        print(
            "Tables selected:",
            len(table_names),
        )

        print(
            ", ".join(table_names)
        )

        for table_index, table_name in enumerate(
            table_names,
            start=1,
        ):
            print()
            print("=" * 100)

            print(
                f"[{table_index}/{len(table_names)}] "
                f"Importing {table_name}"
            )

            try:
                fields = get_table_fields(
                    table_name
                )

                print(
                    "Fields:",
                    len(fields),
                )

                rows = download_table(
                    table_name=table_name,
                    fields=fields,
                )

                table_counts[table_name] = (
                    len(rows)
                )

                filename = (
                    safe_filename(table_name)
                    + ".jsonl"
                )

                write_jsonl(
                    raw_directory / filename,
                    rows,
                )

                normalized_records = [
                    make_catalog_record(
                        table_name,
                        row,
                    )
                    for row in rows
                ]

                write_jsonl(
                    normalized_directory / filename,
                    normalized_records,
                )

                insert_catalog_records(
                    connection,
                    normalized_records,
                )

                if (
                    table_name.casefold()
                    == "recipes"
                ):
                    recipe_rows = rows

                print(
                    f"Imported {len(rows)} rows."
                )

            except Exception as error:
                failures[table_name] = (
                    f"{type(error).__name__}: "
                    f"{error}"
                )

                print(
                    "FAILED:",
                    failures[table_name],
                )

        missing_core_tables = [
            table_name
            for table_name in CORE_TABLES
            if table_counts.get(table_name, 0) == 0
        ]

        if missing_core_tables:
            raise RuntimeError(
                "Core Cargo tables failed or were empty: "
                + ", ".join(missing_core_tables)
            )

        grouped_recipes = (
            build_grouped_recipes(
                recipe_rows
            )
        )

        write_jsonl(
            normalized_directory
            / "recipes_grouped.jsonl",
            grouped_recipes,
        )

        insert_grouped_recipes(
            connection,
            grouped_recipes,
        )

        metadata = {
            "imported_at": imported_at,
            "api_url": API_URL,
            "all_tables": all_tables,
            "table_counts": table_counts,
            "grouped_recipe_entities": len(
                grouped_recipes
            ),
            "failures": failures,
        }

        (
            build_path
            / "import_report.json"
        ).write_text(
            json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

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
                    "imported_at",
                    imported_at,
                ),
                (
                    "api_url",
                    API_URL,
                ),
                (
                    "table_counts",
                    json.dumps(
                        table_counts,
                        ensure_ascii=False,
                    ),
                ),
                (
                    "grouped_recipe_entities",
                    str(len(grouped_recipes)),
                ),
                (
                    "failures",
                    json.dumps(
                        failures,
                        ensure_ascii=False,
                    ),
                ),
            ],
        )

        connection.commit()

        write_attribution(
            build_path
        )

    except Exception:
        connection.close()

        shutil.rmtree(
            build_path,
            ignore_errors=True,
        )

        raise

    else:
        connection.close()

    backup_path = finalize_catalog(
        build_path=build_path,
        replace=replace,
    )

    print()
    print("=" * 100)
    print("IMPORT COMPLETE")
    print("Catalog:", CATALOG_PATH)
    print("Database:", CATALOG_PATH / database_path.name)
    print(
        "Grouped recipe entities:",
        len(grouped_recipes),
    )

    print()
    print("Core table counts:")

    for table_name in CORE_TABLES:
        print(
            f"- {table_name}: "
            f"{table_counts.get(table_name, 0)}"
        )

    print(
        "- All imported tables:",
        len(table_counts),
    )

    if failures:
        print()
        print(
            "Non-core tables that could not "
            "be imported:"
        )

        for table_name, message in failures.items():
            print(
                f"- {table_name}: {message}"
            )

    if backup_path is not None:
        print()
        print(
            "Previous catalog backup:",
            backup_path,
        )

    print()
    print(
        "Existing curated structured data "
        "was not modified:"
    )

    print(
        DATA_ROOT / "structured"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--all-tables",
        action="store_true",
        help=(
            "Import all discoverable Cargo tables, "
            "not only Items/NPCs/Recipes/Drops."
        ),
    )

    parser.add_argument(
        "--replace",
        action="store_true",
        help=(
            "Atomically replace an existing catalog."
        ),
    )

    arguments = parser.parse_args()

    run_import(
        all_tables=arguments.all_tables,
        replace=arguments.replace,
    )


if __name__ == "__main__":
    main()
