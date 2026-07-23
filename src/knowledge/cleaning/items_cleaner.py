
from __future__ import annotations

import json
import re

from collections import Counter
from pathlib import Path
from typing import Any

from src.utils.paths import portable_path, TERRARIA_CLEANED_ROOT, TERRARIA_CATALOG_ROOT

from ..catalog_store import normalize_catalog_name
from .common import (
    normalize_whitespace,
    parse_bool,
    parse_coin_value,
    parse_float,
    parse_int,
    split_caret_list,
    strip_markup,
)


DEFAULT_INPUT_PATH = (
    TERRARIA_CATALOG_ROOT / "normalized" / "Items.jsonl"
)

DEFAULT_OUTPUT_PATH = TERRARIA_CLEANED_ROOT / "Items.jsonl"

DEFAULT_REPORT_PATH = TERRARIA_CLEANED_ROOT / "Items_report.json"


def _is_empty(
    value: Any,
) -> bool:
    return (
        value is None
        or not str(value).strip()
    )


def _normalize_label(
    value: str,
) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        normalize_whitespace(
            value
        ).casefold(),
    ).strip("_")


def _parse_primary_number(
    value: Any,
) -> int | float | None:
    """
    Parse the primary numeric value in an Item field.

    Examples:
        "85" -> 85
        "6.5" -> 6.5
        "13 (10+3) (set)" -> 13
        "16 (0 with Meteor armor)" -> 16

    Only a number at the beginning of the cleaned field
    is accepted. This avoids extracting unrelated
    numbers from explanatory text.
    """
    if _is_empty(value):
        return None

    direct_number = parse_float(
        value
    )

    if direct_number is not None:
        if direct_number.is_integer():
            return int(direct_number)

        return direct_number

    plain = strip_markup(
        value
    )

    if not plain:
        return None

    plain = (
        plain
        .replace(",", "")
        .strip()
    )

    match = re.match(
        r"^([+-]?"
        r"(?:\d+(?:\.\d*)?|\.\d+))",
        plain,
    )

    if not match:
        return None

    number = float(
        match.group(1)
    )

    if number.is_integer():
        return int(number)

    return number


def _parse_optional_number(
    value: Any,
    *,
    field_name: str,
    warnings: list[str],
) -> int | float | None:
    """
    Parse an optional numeric field and add a warning
    only when a non-empty value cannot be interpreted.
    """
    if _is_empty(value):
        return None

    parsed = _parse_primary_number(
        value
    )

    if parsed is None:
        warnings.append(
            f"{field_name}_not_parsed"
        )

    return parsed


def _parse_percentage_number(
    value: Any,
    *,
    field_name: str,
    warnings: list[str],
) -> int | float | None:
    """
    Parse a percentage while retaining percentage
    units.

    Example:
        "70%" -> 70

    This is appropriate for Terraria tool power and
    critical chance fields.
    """
    if _is_empty(value):
        return None

    plain = strip_markup(
        value
    )

    if not plain:
        warnings.append(
            f"{field_name}_not_parsed"
        )

        return None

    match = re.match(
        r"^\s*([+-]?"
        r"(?:\d+(?:\.\d*)?|\.\d+))"
        r"\s*%?",
        plain,
    )

    if not match:
        warnings.append(
            f"{field_name}_not_parsed"
        )

        return None

    number = float(
        match.group(1)
    )

    if number.is_integer():
        return int(number)

    return number


def _parse_platform_number_variants(
    value: Any,
) -> dict[str, int | float]:
    """
    Parse explicit platform/version variants.

    Example:
        9999 (Desktop, Console and Mobile versions)
        / 99 (3DS version)

    becomes:
        {
            "desktop_console_mobile": 9999,
            "3ds": 99,
        }
    """
    plain = strip_markup(
        value
    )

    if not plain:
        return {}

    matches = re.findall(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))"
        r"\s*\(([^)]+)\)",
        plain,
    )

    output: dict[
        str,
        int | float
    ] = {}

    platform_keywords = {
        "desktop",
        "console",
        "mobile",
        "3ds",
        "old-gen",
        "old gen",
        "version",
        "versions",
    }

    for number_text, label_text in matches:
        normalized_label_text = (
            label_text.casefold()
        )

        if not any(
            keyword
            in normalized_label_text
            for keyword in platform_keywords
        ):
            continue

        label = _normalize_label(
            label_text
        )

        if not label:
            continue

        number = float(
            number_text
        )

        if number.is_integer():
            parsed_number: int | float = int(
                number
            )

        else:
            parsed_number = number

        output[label] = parsed_number

    return output



def _parse_set_defense_variants(
    value: Any,
) -> list[int | float]:
    """
    Parse aggregate armor-set defense values.

    Example:
        : 21 / : 32 / : 23 (set)

    becomes:
        [21, 32, 23]

    These values correspond to alternate head-piece
    configurations, so no single value is selected as
    the primary defense.
    """
    plain = strip_markup(
        value
    )

    if not plain:
        return []

    if (
        "(set)" not in plain.casefold()
        or "/" not in plain
    ):
        return []

    number_texts = re.findall(
        r"(?<![\w.])"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
        plain,
    )

    output: list[int | float] = []

    for number_text in number_texts:
        number = float(
            number_text
        )

        parsed_number: int | float

        if number.is_integer():
            parsed_number = int(number)
        else:
            parsed_number = number

        output.append(
            parsed_number
        )

    return output


def _parse_numeric_field(
    value: Any,
    *,
    field_name: str,
    warnings: list[str],
) -> dict[str, Any]:
    """
    Preserve the primary numeric value, platform
    variants, aggregate variants, and semantic labels.

    Special cases:
        rare="quest"
        armor-set defense with several head-piece
        configurations
    """
    primary = None
    semantic_label = None
    variants: list[int | float] = []

    if not _is_empty(value):
        primary = _parse_primary_number(
            value
        )

        plain = strip_markup(
            value
        )

        normalized_plain = (
            plain.casefold().strip()
            if plain
            else ""
        )

        if (
            primary is None
            and field_name == "rarity"
            and normalized_plain == "quest"
        ):
            semantic_label = "quest"

        if (
            primary is None
            and field_name == "defense"
        ):
            variants = (
                _parse_set_defense_variants(
                    value
                )
            )

        if (
            primary is None
            and semantic_label is None
            and not variants
        ):
            warnings.append(
                f"{field_name}_not_parsed"
            )

    return {
        "primary": primary,
        "semantic_label": semantic_label,
        "variants": variants,
        "by_platform": (
            _parse_platform_number_variants(
                value
            )
        ),
        "raw_value": value,
    }


def _parse_knockback(
    value: Any,
    *,
    warnings: list[str],
) -> tuple[int | float | None, str | None]:
    """
    Parse either numeric knockback or Terraria's
    textual knockback classification.

    Examples:
        6.5               -> (6.5, None)
        Average knockback -> (None, "average")
    """
    if _is_empty(value):
        return None, None

    numeric_value = (
        _parse_primary_number(
            value
        )
    )

    if numeric_value is not None:
        return numeric_value, None

    plain = strip_markup(
        value
    )

    if plain:
        match = re.fullmatch(
            r"\s*("
            r"none|"
            r"extremely weak|"
            r"very weak|"
            r"weak|"
            r"average|"
            r"strong|"
            r"very strong|"
            r"extremely strong|"
            r"insane"
            r")\s+knockback\s*",
            plain,
            flags=re.IGNORECASE,
        )

        if match:
            return (
                None,
                match.group(1).casefold(),
            )

    warnings.append(
        "knockback_not_parsed"
    )

    return None, None

def _extract_coin_sort_values(
    value: Any,
) -> list[int]:
    if _is_empty(value):
        return []

    values = [
        int(match)
        for match in re.findall(
            r'data-sort-value\s*=\s*'
            r'["\'](\d+)["\']',
            str(value),
            flags=re.IGNORECASE,
        )
    ]

    unique_values: list[int] = []

    for coin_value in values:
        if coin_value not in unique_values:
            unique_values.append(
                coin_value
            )

    return unique_values



def _parse_alternate_currency(
    value: Any,
) -> dict[str, Any] | None:
    """
    Parse supported non-coin currencies.

    Example:
        title="45 Defender Medals"

    becomes:
        {
            "currency": "defender_medal",
            "display_name": "Defender Medal",
            "amount": 45,
        }
    """
    if _is_empty(value):
        return None

    raw_text = str(value)

    match = re.search(
        r'title\s*=\s*["\']'
        r"\s*(\d+)\s+"
        r"Defender\s+Medals?"
        r'\s*["\']',
        raw_text,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    return {
        "currency": "defender_medal",
        "display_name": (
            "Defender Medal"
        ),
        "amount": int(
            match.group(1)
        ),
    }


def _parse_price(
    value: Any,
    *,
    field_name: str,
    warnings: list[str],
) -> dict[str, Any]:
    """
    Parse copper prices, supported alternate
    currencies, and explicit no-value markers.
    """
    if _is_empty(value):
        return {
            "primary_copper": None,
            "detected_copper_values": [],
            "alternate_currency": None,
            "explicit_no_value": False,
            "has_variants": False,
            "plain_text": None,
            "raw_value": value,
        }

    plain_text = strip_markup(
        value
    )

    normalized_plain = (
        plain_text.casefold().strip()
        if plain_text
        else ""
    )

    explicit_no_value = (
        normalized_plain
        in {
            "n/a",
            "na",
            "none",
            "no value",
            "not applicable",
        }
    )

    detected_values = (
        _extract_coin_sort_values(
            value
        )
    )

    primary_copper = (
        detected_values[0]
        if detected_values
        else parse_coin_value(value)
    )

    alternate_currency = (
        _parse_alternate_currency(
            value
        )
    )

    if (
        primary_copper is None
        and alternate_currency is None
        and not explicit_no_value
    ):
        warnings.append(
            f"{field_name}_not_parsed"
        )

    return {
        "primary_copper": primary_copper,

        "detected_copper_values": (
            detected_values
        ),

        "alternate_currency": (
            alternate_currency
        ),

        "explicit_no_value": (
            explicit_no_value
        ),

        "has_variants": (
            len(detected_values) > 1
        ),

        "plain_text": plain_text,
        "raw_value": value,
    }

def _clean_damage_type(
    value: Any,
) -> str | None:
    cleaned = strip_markup(
        value
    )

    if not cleaned:
        return None

    return cleaned.casefold()


def _clean_text_list(
    value: Any,
) -> list[str]:
    return split_caret_list(
        value
    )


def clean_item_record(
    record: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert one normalized Item record into a stable
    cleaned Item record.
    """
    if not isinstance(record, dict):
        raise TypeError(
            "Item record must be a dictionary."
        )

    data = record.get("data")

    if not isinstance(data, dict):
        raise ValueError(
            "Normalized Item record is missing "
            "its data dictionary."
        )

    warnings: list[str] = []

    name = (
        strip_markup(data.get("name"))
        or strip_markup(
            record.get("entity_name")
        )
        or strip_markup(
            data.get("_pageName")
        )
    )

    if not name:
        name = "Unknown Item"
        warnings.append(
            "missing_item_name"
        )

    item_id = parse_int(
        data.get("itemid")
        or record.get("entity_id")
    )

    # Missing item IDs commonly indicate aggregate
    # set pages, legacy entries, or other non-standard
    # catalog records. Preserve item_id=None and track
    # it in the report, but do not treat it as a parse
    # failure.
    internal_name = strip_markup(
        data.get("internalname")
    )

    stack = _parse_numeric_field(
        data.get("stack"),
        field_name="stack",
        warnings=warnings,
    )

    rarity = _parse_numeric_field(
        data.get("rare"),
        field_name="rarity",
        warnings=warnings,
    )

    research_count = (
        _parse_optional_number(
            data.get("research"),
            field_name="research",
            warnings=warnings,
        )
    )

    damage = _parse_optional_number(
        data.get("damage"),
        field_name="damage",
        warnings=warnings,
    )

    defense = _parse_numeric_field(
        data.get("defense"),
        field_name="defense",
        warnings=warnings,
    )

    velocity = _parse_optional_number(
        data.get("velocity"),
        field_name="velocity",
        warnings=warnings,
    )

    (
        knockback,
        knockback_label,
    ) = _parse_knockback(
        data.get("knockback"),
        warnings=warnings,
    )

    use_time = _parse_numeric_field(
        data.get("usetime"),
        field_name="use_time",
        warnings=warnings,
    )

    mana = _parse_numeric_field(
        data.get("mana"),
        field_name="mana",
        warnings=warnings,
    )

    critical_percent = (
        _parse_percentage_number(
            data.get("critical"),
            field_name="critical",
            warnings=warnings,
        )
    )

    axe_power = _parse_percentage_number(
        data.get("axe"),
        field_name="axe",
        warnings=warnings,
    )

    pickaxe_power = _parse_percentage_number(
        data.get("pick"),
        field_name="pick",
        warnings=warnings,
    )

    hammer_power = _parse_percentage_number(
        data.get("hammer"),
        field_name="hammer",
        warnings=warnings,
    )

    fishing_power = (
        _parse_optional_number(
            data.get("fishing"),
            field_name="fishing",
            warnings=warnings,
        )
    )

    bait_power = _parse_optional_number(
        data.get("bait"),
        field_name="bait",
        warnings=warnings,
    )

    fishing_bonus = (
        _parse_optional_number(
            data.get("bonus"),
            field_name="bonus",
            warnings=warnings,
        )
    )

    tool_speed = _parse_optional_number(
        data.get("toolspeed"),
        field_name="tool_speed",
        warnings=warnings,
    )

    health_heal = (
        _parse_optional_number(
            data.get("hheal"),
            field_name="health_heal",
            warnings=warnings,
        )
    )

    mana_heal = _parse_optional_number(
        data.get("mheal"),
        field_name="mana_heal",
        warnings=warnings,
    )

    placed_width = (
        _parse_optional_number(
            data.get("placedwidth"),
            field_name="placed_width",
            warnings=warnings,
        )
    )

    placed_height = (
        _parse_optional_number(
            data.get("placedheight"),
            field_name="placed_height",
            warnings=warnings,
        )
    )

    buy_price = _parse_price(
        data.get("buy"),
        field_name="buy_price",
        warnings=warnings,
    )

    sell_price = _parse_price(
        data.get("sell"),
        field_name="sell_price",
        warnings=warnings,
    )

    return {
        "source_catalog_id": record.get(
            "catalog_id"
        ),
        "source_row_id": data.get("_rowID"),
        "source_page": data.get("_pageName"),

        "record_type": "item",

        "name": name,
        "normalized_name": (
            normalize_catalog_name(
                name
            )
        ),
        "item_id": item_id,
        "has_item_id": item_id is not None,
        "internal_name": internal_name,

        "images": {
            "item_file": (
                data.get("imagefile")
                or None
            ),
            "placed_raw": (
                data.get("imageplaced")
                or None
            ),
            "equipped_raw": (
                data.get("imageequipped")
                or None
            ),
        },

        "classification": {
            "types": _clean_text_list(
                data.get("type")
            ),
            "list_categories": (
                _clean_text_list(
                    data.get("listcat")
                )
            ),
            "tags": _clean_text_list(
                data.get("tag")
            ),
        },

        "flags": {
            "autoswing": parse_bool(
                data.get("autoswing"),
                empty_value=False,
            ),
            "consumable": parse_bool(
                data.get("consumable"),
                empty_value=False,
            ),
            "hardmode": parse_bool(
                data.get("hardmode"),
                empty_value=False,
            ),
            "unobtainable": parse_bool(
                data.get("unobtainable"),
                empty_value=False,
            ),
            "placeable": parse_bool(
                data.get("placeable"),
                empty_value=False,
            ),
        },

        "inventory": {
            "stack_limit": stack,
            "research_count": research_count,
            "rarity": rarity,
        },

        "combat": {
            "damage": damage,
            "damage_type": (
                _clean_damage_type(
                    data.get("damagetype")
                )
            ),
            "critical_chance_percent": (
                critical_percent
            ),
            "defense": defense,
            "velocity": velocity,
            "knockback": knockback,
            "knockback_label": (
                knockback_label
            ),
            "use_time": use_time,
            "mana_cost": mana,
        },

        "tools": {
            "axe_power_percent": axe_power,
            "pickaxe_power_percent": (
                pickaxe_power
            ),
            "hammer_power_percent": (
                hammer_power
            ),
            "fishing_power": fishing_power,
            "bait_power": bait_power,
            "fishing_bonus": fishing_bonus,
            "tool_speed": tool_speed,
        },

        "restoration": {
            "health": health_heal,
            "mana": mana_heal,
        },

        "placement": {
            "width": placed_width,
            "height": placed_height,
        },

        "equipment": {
            "body_slot": strip_markup(
                data.get("bodyslot")
            ),
        },

        "effects": {
            "buffs": _clean_text_list(
                data.get("buffs")
            ),
            "debuffs": _clean_text_list(
                data.get("debuffs")
            ),
        },

        "economy": {
            "buy": buy_price,
            "sell": sell_price,
        },

        "tooltip": strip_markup(
            data.get("tooltip")
        ),

        "raw_fields": {
            "stack": data.get("stack"),
            "rare": data.get("rare"),
            "damage": data.get("damage"),
            "defense": data.get("defense"),
            "velocity": data.get("velocity"),
            "knockback": data.get(
                "knockback"
            ),
            "usetime": data.get("usetime"),
            "mana": data.get("mana"),
            "buy": data.get("buy"),
            "sell": data.get("sell"),
        },

        "parse_status": (
            "ok"
            if not warnings
            else "partial"
        ),

        "parse_warnings": warnings,
    }


def clean_items_file(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    """
    Clean all Items records and write JSONL plus a
    summary report.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    report_path = Path(report_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Items input file not found: "
            f"{input_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_output_path = (
        output_path.with_suffix(
            output_path.suffix + ".tmp"
        )
    )

    total_records = 0
    ok_records = 0
    partial_records = 0

    missing_item_id_records = 0
    buy_price_parsed_records = 0
    sell_price_parsed_records = 0
    platform_variant_records = 0

    warning_counts: Counter[str] = (
        Counter()
    )

    seen_catalog_ids: set[str] = set()

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

            cleaned = clean_item_record(
                record
            )

            source_catalog_id = cleaned.get(
                "source_catalog_id"
            )

            if not source_catalog_id:
                raise ValueError(
                    "Missing source_catalog_id at "
                    f"{input_path}:{line_number}"
                )

            if source_catalog_id in seen_catalog_ids:
                raise ValueError(
                    "Duplicate source_catalog_id: "
                    f"{source_catalog_id}"
                )

            seen_catalog_ids.add(
                source_catalog_id
            )

            total_records += 1

            if cleaned["parse_status"] == "ok":
                ok_records += 1
            else:
                partial_records += 1

            if cleaned["item_id"] is None:
                missing_item_id_records += 1

            if cleaned[
                "economy"
            ]["buy"]["primary_copper"] is not None:
                buy_price_parsed_records += 1

            if cleaned[
                "economy"
            ]["sell"]["primary_copper"] is not None:
                sell_price_parsed_records += 1

            has_platform_variants = any(
                bool(field["by_platform"])
                for field in [
                    cleaned[
                        "inventory"
                    ]["stack_limit"],
                    cleaned[
                        "inventory"
                    ]["rarity"],
                    cleaned[
                        "combat"
                    ]["defense"],
                    cleaned[
                        "combat"
                    ]["use_time"],
                    cleaned[
                        "combat"
                    ]["mana_cost"],
                ]
            )

            if has_platform_variants:
                platform_variant_records += 1

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
            "Items input file contained "
            "no records."
        )

    temporary_output_path.replace(
        output_path
    )

    report = {
        "input_path": portable_path(input_path),
        "output_path": portable_path(output_path),

        "total_records": total_records,
        "ok_records": ok_records,
        "partial_records": partial_records,

        "unique_source_catalog_ids": len(
            seen_catalog_ids
        ),

        "missing_item_id_records": (
            missing_item_id_records
        ),

        "buy_price_parsed_records": (
            buy_price_parsed_records
        ),

        "sell_price_parsed_records": (
            sell_price_parsed_records
        ),

        "platform_variant_records": (
            platform_variant_records
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
