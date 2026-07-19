
from __future__ import annotations

import json
import re

from collections import Counter
from pathlib import Path
from typing import Any

from ..catalog_store import normalize_catalog_name
from .common import (
    parse_coin_value,
    parse_mode_coin_values,
    parse_int,
    parse_labeled_numbers,
    parse_mode_numbers,
    split_caret_list,
    strip_markup,
)


DEFAULT_INPUT_PATH = Path(
    "/content/llm_project/data/terraria/"
    "catalog/normalized/NPCs.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "/content/llm_project/data/terraria/"
    "catalog/cleaned/NPCs.jsonl"
)

DEFAULT_REPORT_PATH = Path(
    "/content/llm_project/data/terraria/"
    "catalog/cleaned/NPCs_report.json"
)


def _clean_optional_text(
    value: Any,
) -> str | None:
    """
    Convert optional HTML/Wiki text into plain text.
    """
    return strip_markup(value)


def _clean_environment(
    value: Any,
) -> list[str]:
    """
    Convert NPC environment into a list.

    Cargo mainly uses '^' for multiple values, but
    some records contain one ordinary string.
    """
    return split_caret_list(value)


def _clean_npc_types(
    value: Any,
) -> list[str]:
    """
    Convert an NPC type such as:

        boss Part
        town NPC
        enemy

    into normalized type tags.
    """
    cleaned = strip_markup(value)

    if not cleaned:
        return []

    text = cleaned.casefold()

    known_tags = []

    tag_patterns = [
        ("boss", r"\bboss\b"),
        ("boss_part", r"\bboss\s+part\b"),
        ("town_npc", r"\btown\s+npc\b"),
        ("critter", r"\bcritter\b"),
        ("enemy", r"\benemy\b"),
        ("projectile", r"\bprojectile\b"),
        ("servant", r"\bservant\b"),
        ("worm", r"\bworm\b"),
        ("part", r"\bpart\b"),
    ]

    for tag, pattern in tag_patterns:
        if re.search(pattern, text):
            known_tags.append(tag)

    if known_tags:
        # boss_part already implies boss and part.
        if "boss_part" in known_tags:
            known_tags = [
                tag
                for tag in known_tags
                if tag not in {
                    "boss",
                    "part",
                }
            ]

        return known_tags

    # Preserve unknown type values instead of dropping them.
    fallback = re.sub(
        r"[^a-z0-9]+",
        "_",
        text,
    ).strip("_")

    return [fallback] if fallback else []


def _collapse_mode_values(
    values: dict[str, list[int | float]],
) -> dict[str, int | float | list[int | float]]:
    """
    Collapse one-element lists into scalar values.

    Life and defense usually have one number per mode,
    while damage may contain multiple attack values.
    """
    output: dict[
        str,
        int | float | list[int | float]
    ] = {}

    for mode, mode_values in values.items():
        if len(mode_values) == 1:
            output[mode] = mode_values[0]
        else:
            output[mode] = mode_values

    return output


def _parse_knockback(
    value: Any,
) -> dict[str, Any]:
    """
    Parse knockback-resistance percentages.

    The Wiki field commonly contains values such as
    100%. Values are retained as percentages because
    this avoids ambiguity over whether 100 means
    immunity or a multiplier.
    """
    mode_values = parse_mode_numbers(value)

    percentages = _collapse_mode_values(
        mode_values
    )

    return {
        "percent": percentages,
        "raw_value": value,
    }



def _parse_money(
    value: Any,
    warnings: list[str],
) -> dict[str, Any]:
    """
    Parse NPC coin drops by difficulty mode and,
    when present, progression stage.

    Multiple valid values are no longer considered
    an error. They are represented explicitly in
    by_mode.
    """
    if value is None or not str(value).strip():
        return {
            "normal_copper": None,
            "by_mode": {},
            "raw_value": value,
        }

    by_mode = parse_mode_coin_values(
        value
    )

    normal_copper = None

    normal_values = by_mode.get(
        "normal",
        {},
    )

    normal_default = normal_values.get(
        "default"
    )

    if isinstance(normal_default, int):
        normal_copper = normal_default

    elif isinstance(normal_default, list):
        if normal_default:
            normal_copper = normal_default[0]

    if normal_copper is None:
        all_values = by_mode.get(
            "all",
            {},
        )

        all_default = all_values.get(
            "default"
        )

        if isinstance(all_default, int):
            normal_copper = all_default

        elif isinstance(all_default, list):
            if all_default:
                normal_copper = all_default[0]

    if normal_copper is None:
        normal_copper = parse_coin_value(
            value
        )

    if not by_mode and normal_copper is None:
        warnings.append(
            "money_not_parsed"
        )

    return {
        "normal_copper": normal_copper,
        "by_mode": by_mode,
        "raw_value": value,
    }


def clean_npc_record(
    record: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert one normalized NPC Cargo record into a
    stable cleaned NPC record.
    """
    if not isinstance(record, dict):
        raise TypeError(
            "NPC record must be a dictionary."
        )

    data = record.get("data")

    if not isinstance(data, dict):
        raise ValueError(
            "Normalized NPC record is missing "
            "its data dictionary."
        )

    warnings: list[str] = []

    # Always prefer nameraw. Some older normalized files
    # contain HTML in top-level entity_name.
    name = (
        strip_markup(data.get("nameraw"))
        or strip_markup(record.get("entity_name"))
        or strip_markup(data.get("_pageName"))
    )

    if not name:
        warnings.append("missing_entity_name")
        name = "Unknown NPC"

    normalized_name = normalize_catalog_name(
        name
    )

    npc_id = parse_int(
        data.get("npcid")
        or record.get("entity_id")
    )

    if npc_id is None:
        warnings.append("missing_or_invalid_npc_id")

    life_raw = data.get("life")
    damage_raw = data.get("damage")
    defense_raw = data.get("defense")
    knockback_raw = data.get("knockback")

    life = _collapse_mode_values(
        parse_mode_numbers(life_raw)
    )

    damage = _collapse_mode_values(
        parse_mode_numbers(damage_raw)
    )

    defense = _collapse_mode_values(
        parse_mode_numbers(defense_raw)
    )

    # Some objects, such as the Eternia Crystal, use
    # event tiers instead of difficulty-mode markup:
    # 14 (T1) / 18 (T2) / 20 (T3)
    if defense_raw and not defense:
        tier_values = parse_labeled_numbers(
            defense_raw
        )

        if tier_values:
            defense = {
                "tiers": tier_values,
            }

    if life_raw and not life:
        warnings.append("life_not_parsed")

    if damage_raw and not damage:
        warnings.append("damage_not_parsed")

    if defense_raw and not defense:
        warnings.append("defense_not_parsed")

    immunities = split_caret_list(
        data.get("immunities")
    )

    banner_name = (
        strip_markup(data.get("bannername"))
        or strip_markup(data.get("banner"))
    )

    cleaned_record = {
        "source_catalog_id": record.get(
            "catalog_id"
        ),
        "source_row_id": data.get("_rowID"),
        "source_page": data.get("_pageName"),

        "record_type": "npc",

        "name": name,
        "normalized_name": normalized_name,
        "npc_id": npc_id,

        "npc_types": _clean_npc_types(
            data.get("type")
        ),

        "environment": _clean_environment(
            data.get("environment")
        ),

        "ai": _clean_optional_text(
            data.get("ai")
        ),

        "stats": {
            "life": life,
            "damage": damage,
            "defense": defense,
            "knockback_resistance": (
                _parse_knockback(
                    knockback_raw
                )
            ),
        },

        "immunities": immunities,

        "banner_name": banner_name,

        "coin_drop": _parse_money(
            data.get("money"),
            warnings,
        ),

        # Keep only the raw fields needed to diagnose
        # parsers. The complete source remains in
        # normalized/NPCs.jsonl.
        "raw_fields": {
            "type": data.get("type"),
            "environment": data.get(
                "environment"
            ),
            "life": life_raw,
            "damage": damage_raw,
            "defense": defense_raw,
            "knockback": knockback_raw,
            "money": data.get("money"),
            "immunities": data.get(
                "immunities"
            ),
        },

        "parse_status": (
            "ok"
            if not warnings
            else "partial"
        ),

        "parse_warnings": warnings,
    }

    return cleaned_record


def clean_npc_file(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    """
    Clean every NPC record and write JSONL output.

    The output is written to a temporary file first,
    then atomically replaces the destination.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    report_path = Path(report_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"NPC input file not found: {input_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_output_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    total_records = 0
    ok_records = 0
    partial_records = 0
    warning_counts: Counter[str] = Counter()

    seen_source_ids: set[str] = set()

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as input_file, temporary_output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        for line_number, line in enumerate(
            input_file,
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
                    f"{input_path}:{line_number}"
                ) from error

            cleaned = clean_npc_record(
                record
            )

            source_catalog_id = cleaned.get(
                "source_catalog_id"
            )

            if not source_catalog_id:
                raise ValueError(
                    f"Missing source_catalog_id at "
                    f"{input_path}:{line_number}"
                )

            if source_catalog_id in seen_source_ids:
                raise ValueError(
                    "Duplicate source_catalog_id: "
                    f"{source_catalog_id}"
                )

            seen_source_ids.add(
                source_catalog_id
            )

            total_records += 1

            if cleaned["parse_status"] == "ok":
                ok_records += 1
            else:
                partial_records += 1

            warning_counts.update(
                cleaned["parse_warnings"]
            )

            output_file.write(
                json.dumps(
                    cleaned,
                    ensure_ascii=False,
                )
                + "\n"
            )

    if total_records == 0:
        temporary_output_path.unlink(
            missing_ok=True
        )

        raise ValueError(
            "NPC input file contained no records."
        )

    temporary_output_path.replace(
        output_path
    )

    report = {
        "input_path": str(input_path),
        "output_path": str(output_path),

        "total_records": total_records,
        "ok_records": ok_records,
        "partial_records": partial_records,

        "unique_source_catalog_ids": len(
            seen_source_ids
        ),

        "warning_counts": dict(
            warning_counts.most_common()
        ),
    }

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
