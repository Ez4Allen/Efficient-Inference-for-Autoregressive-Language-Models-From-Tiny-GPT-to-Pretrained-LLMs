"""Build the compact Stardew Valley fact database."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from src.utils.io import read_jsonl, write_json
from src.utils.paths import STARDEW_CATALOG_ROOT, portable_path

from .normalizers import SEASONS, normalize_name, parse_clock

DEFAULT_FACTS_PATH = STARDEW_CATALOG_ROOT / "cleaned" / "facts.jsonl"
DEFAULT_DATABASE_PATH = STARDEW_CATALOG_ROOT / "stardew_query.sqlite3"
DEFAULT_REPORT_PATH = STARDEW_CATALOG_ROOT / "reports" / "build_report.json"


ALLOWED_RECORD_TYPES = {"crop", "fish", "villager", "recipe", "bundle"}
ALLOWED_PLATFORMS = {"all", "pc", "console", "mobile", "legacy", "unknown"}
ALLOWED_PARSE_STATUSES = {"ok", "partial"}
ALLOWED_BUNDLE_MODES = {"standard", "remixed", "missing_bundle"}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_record(record: dict[str, Any]) -> None:
    required = {
        "schema_version", "game", "game_version", "platform", "record_type",
        "source_catalog_id", "name", "normalized_name", "aliases", "facts",
        "conditions", "provenance", "parse_status", "parse_warnings",
    }
    missing = required - record.keys()
    if missing:
        raise ValueError(f"Stardew record is missing fields: {sorted(missing)}")
    if record["schema_version"] != 1:
        raise ValueError("Unsupported Stardew record schema_version.")
    if record["game"] != "stardew_valley":
        raise ValueError("Stardew records must use game='stardew_valley'.")
    if record["record_type"] not in ALLOWED_RECORD_TYPES:
        raise ValueError(f"Unsupported Stardew record_type: {record['record_type']!r}.")
    if record["platform"] not in ALLOWED_PLATFORMS:
        raise ValueError(f"Unsupported Stardew platform: {record['platform']!r}.")
    if record["parse_status"] not in ALLOWED_PARSE_STATUSES:
        raise ValueError(f"Unsupported parse_status: {record['parse_status']!r}.")
    if not str(record["game_version"]).strip():
        raise ValueError("game_version cannot be empty.")
    if not str(record["source_catalog_id"]).strip():
        raise ValueError("source_catalog_id cannot be empty.")
    if not str(record["name"]).strip():
        raise ValueError("name cannot be empty.")
    if normalize_name(record["name"]) != record["normalized_name"]:
        raise ValueError(f"normalized_name mismatch for {record['name']!r}.")
    aliases = record.get("aliases")
    if not isinstance(aliases, list):
        raise TypeError(f"aliases must be a list: {record['source_catalog_id']}.")
    if any(not str(alias).strip() for alias in aliases):
        raise ValueError(f"Empty alias in {record['source_catalog_id']}.")
    if not isinstance(record.get("facts"), dict) or not isinstance(record.get("conditions"), dict):
        raise TypeError("facts and conditions must be mappings.")
    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        raise TypeError("provenance must be a mapping.")
    for field in ("source_name", "page_title", "source_url", "license_name"):
        if not str(provenance.get(field, "")).strip():
            raise ValueError(
                f"Missing provenance field {field!r}: {record['source_catalog_id']}"
            )

    facts = record.get("facts") or {}
    if record["record_type"] == "crop":
        growth = facts.get("growth_days")
        if not isinstance(growth, int) or growth <= 0:
            raise ValueError(f"Invalid crop growth_days: {record['name']}")
        for season in facts.get("seasons") or []:
            if season not in SEASONS:
                raise ValueError(f"Invalid crop season {season!r}.")
        quantity = facts.get("harvest_quantity") or {}
        if float(quantity.get("minimum", 0)) <= 0:
            raise ValueError(f"Invalid crop harvest quantity: {record['name']}")
        regrow = facts.get("regrow_days")
        if regrow is not None and (not isinstance(regrow, int) or regrow <= 0):
            raise ValueError(f"Invalid crop regrow_days: {record['name']}")
    elif record["record_type"] == "fish":
        windows = facts.get("availability_windows") or []
        if not windows:
            raise ValueError(f"Fish has no availability window: {record['name']}")
        for window in windows:
            start = parse_clock(window.get("time_start"))
            end = parse_clock(window.get("time_end"))
            if start is None or end is None or end < start:
                raise ValueError(f"Invalid fish time window: {record['name']}")
            for season in window.get("seasons") or []:
                if season not in SEASONS:
                    raise ValueError(f"Invalid fish season {season!r}.")
            if not window.get("weather") or not window.get("locations"):
                raise ValueError(f"Incomplete fish availability window: {record['name']}")
    elif record["record_type"] == "villager":
        birthday = facts.get("birthday") or {}
        if birthday.get("season") not in SEASONS:
            raise ValueError(f"Invalid villager birthday season: {record['name']}")
        day = birthday.get("day")
        if not isinstance(day, int) or not 1 <= day <= 28:
            raise ValueError(f"Invalid villager birthday day: {record['name']}")
        if not isinstance(facts.get("loved_gifts"), list):
            raise TypeError(f"loved_gifts must be a list: {record['name']}")
    elif record["record_type"] == "recipe":
        ingredients = facts.get("ingredients") or []
        if not ingredients:
            raise ValueError(f"Recipe has no ingredients: {record['name']}")
        for ingredient in ingredients:
            if float(ingredient.get("quantity", 0)) <= 0:
                raise ValueError(f"Invalid ingredient quantity: {record['name']}")
    elif record["record_type"] == "bundle":
        if facts.get("bundle_mode") not in ALLOWED_BUNDLE_MODES:
            raise ValueError(f"Invalid bundle mode: {record['name']}")
        requirements = facts.get("requirements") or []
        if not requirements:
            raise ValueError(f"Bundle has no requirements: {record['name']}")
        for requirement in requirements:
            if float(requirement.get("quantity", 0)) <= 0:
                raise ValueError(f"Invalid bundle quantity: {record['name']}")


def build_stardew_database(
    *,
    facts_path: str | Path = DEFAULT_FACTS_PATH,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    facts_path = Path(facts_path).expanduser().resolve()
    database_path = Path(database_path).expanduser().resolve()
    report_path = Path(report_path).expanduser().resolve()
    records = read_jsonl(facts_path)
    if not records:
        raise ValueError("The Stardew fact snapshot is empty.")

    ids: set[str] = set()
    for record in records:
        validate_record(record)
        record_id = record["source_catalog_id"]
        if record_id in ids:
            raise ValueError(f"Duplicate Stardew record ID: {record_id}")
        ids.add(record_id)

    database_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = database_path.with_suffix(database_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = OFF")
        connection.execute("PRAGMA synchronous = OFF")
        connection.executescript(
            """
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;

            CREATE TABLE records (
                source_catalog_id TEXT PRIMARY KEY,
                record_type TEXT NOT NULL,
                name TEXT NOT NULL,
                normalized_name TEXT NOT NULL,
                game_version TEXT NOT NULL,
                platform TEXT NOT NULL,
                parse_status TEXT NOT NULL,
                facts_json TEXT NOT NULL,
                conditions_json TEXT NOT NULL,
                provenance_json TEXT NOT NULL,
                record_json TEXT NOT NULL
            ) WITHOUT ROWID;

            CREATE TABLE aliases (
                normalized_alias TEXT NOT NULL,
                alias TEXT NOT NULL,
                source_catalog_id TEXT NOT NULL,
                PRIMARY KEY(normalized_alias, source_catalog_id),
                FOREIGN KEY(source_catalog_id) REFERENCES records(source_catalog_id)
            ) WITHOUT ROWID;

            CREATE INDEX idx_records_type_name ON records(record_type, normalized_name);
            CREATE INDEX idx_aliases_target ON aliases(source_catalog_id);

            CREATE VIRTUAL TABLE records_fts USING fts5(
                source_catalog_id UNINDEXED,
                name,
                aliases,
                record_type,
                tokenize='unicode61 remove_diacritics 2'
            );
            """
        )
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [
                ("schema_version", "1"),
                ("database_type", "stardew_fact_catalog"),
                ("source_sha256", _sha256(facts_path)),
                ("facts_path", portable_path(facts_path)),
            ],
        )
        connection.executemany(
            "INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row["source_catalog_id"], row["record_type"], row["name"],
                    row["normalized_name"], row["game_version"], row["platform"],
                    row["parse_status"], _json(row["facts"]), _json(row["conditions"]),
                    _json(row["provenance"]), _json(row),
                )
                for row in records
            ],
        )
        alias_rows = []
        for row in records:
            aliases = [row["name"], *(row.get("aliases") or [])]
            seen: set[str] = set()
            for alias in aliases:
                normalized = normalize_name(alias)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                alias_rows.append((normalized, str(alias), row["source_catalog_id"]))
        connection.executemany(
            "INSERT INTO aliases VALUES (?, ?, ?)",
            alias_rows,
        )
        connection.executemany(
            "INSERT INTO records_fts VALUES (?, ?, ?, ?)",
            [
                (
                    row["source_catalog_id"], row["name"],
                    " ".join(row.get("aliases") or []), row["record_type"],
                )
                for row in records
            ],
        )
        connection.commit()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        foreign_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
        if integrity is None or integrity[0] != "ok":
            raise AssertionError(f"SQLite integrity check failed: {integrity}")
        if foreign_errors:
            raise AssertionError(f"Stardew foreign-key errors: {foreign_errors[:5]}")
    except Exception:
        connection.close()
        temporary.unlink(missing_ok=True)
        raise
    else:
        connection.close()
    temporary.replace(database_path)

    report = {
        "status": "passed",
        "facts_path": portable_path(facts_path),
        "database_path": portable_path(database_path),
        "record_count": len(records),
        "record_type_counts": dict(Counter(row["record_type"] for row in records)),
        "alias_count": len(alias_rows),
        "integrity_check": "ok",
        "foreign_key_errors": 0,
        "database_size_bytes": database_path.stat().st_size,
        "source_sha256": _sha256(facts_path),
    }
    write_json(report_path, report)
    return report
