
from __future__ import annotations

import html
import json
import re

from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from ..catalog_store import normalize_catalog_name
from .common import (
    normalize_whitespace,
    parse_bool,
    parse_int,
    strip_markup,
)


DEFAULT_INPUT_PATH = Path(
    "/content/llm_project/data/terraria/"
    "catalog/normalized/Drops.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "/content/llm_project/data/terraria/"
    "catalog/cleaned/Drops.jsonl"
)

DEFAULT_REPORT_PATH = Path(
    "/content/llm_project/data/terraria/"
    "catalog/cleaned/Drops_report.json"
)


_MODE_NAMES = (
    "normal",
    "expert",
    "master",
)


def _normalize_label(
    value: str,
) -> str:
    value = html.unescape(value)
    value = normalize_whitespace(value)

    return re.sub(
        r"[^a-z0-9]+",
        "_",
        value.casefold(),
    ).strip("_")


def _enabled_modes(
    data: dict[str, Any],
) -> list[str]:
    """
    Read Cargo's normal/expert/master availability
    flags.
    """
    enabled: list[str] = []

    for mode in _MODE_NAMES:
        value = parse_bool(
            data.get(mode),
            empty_value=False,
        )

        if value:
            enabled.append(mode)

    return enabled


class _ModeValueParser(HTMLParser):
    """
    Collect visible text according to Terraria mode
    CSS classes.

    Supports:
        m-normal
        m-expert
        m-master
        m-expert-master
        m-all
        mode-exclusive expert/master
    """

    def __init__(self) -> None:
        super().__init__()

        self.mode_stack: list[set[str]] = []

        self.text_by_mode: dict[
            str,
            list[str]
        ] = {
            "all": [],
            "normal": [],
            "expert": [],
            "master": [],
        }

    @staticmethod
    def _modes_from_classes(
        class_tokens: set[str],
    ) -> set[str]:
        modes: set[str] = set()

        if "m-all" in class_tokens:
            modes.add("all")

        if (
            "m-normal" in class_tokens
            or "m-journey" in class_tokens
        ):
            modes.add("normal")

        if "m-expert-master" in class_tokens:
            modes.update(
                {
                    "expert",
                    "master",
                }
            )

        if "m-expert" in class_tokens:
            modes.add("expert")

        if "m-master" in class_tokens:
            modes.add("master")

        if (
            "expert" in class_tokens
            and "mode-exclusive" in class_tokens
            and "m-expert-master"
            not in class_tokens
        ):
            modes.add("expert")

        if (
            "master" in class_tokens
            and "mode-exclusive" in class_tokens
        ):
            modes.add("master")

        return modes

    def handle_starttag(
        self,
        tag: str,
        attrs: list[
            tuple[str, str | None]
        ],
    ) -> None:
        attribute_map = {
            key: value or ""
            for key, value in attrs
        }

        class_tokens = set(
            attribute_map.get(
                "class",
                "",
            ).split()
        )

        own_modes = self._modes_from_classes(
            class_tokens
        )

        inherited_modes = (
            self.mode_stack[-1]
            if self.mode_stack
            else set()
        )

        active_modes = (
            own_modes
            if own_modes
            else inherited_modes
        )

        self.mode_stack.append(
            set(active_modes)
        )

    def handle_endtag(
        self,
        tag: str,
    ) -> None:
        if self.mode_stack:
            self.mode_stack.pop()

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[
            tuple[str, str | None]
        ],
    ) -> None:
        return

    def handle_data(
        self,
        data: str,
    ) -> None:
        text = normalize_whitespace(data)

        if not text:
            return

        active_modes = (
            self.mode_stack[-1]
            if self.mode_stack
            else set()
        )

        for mode in active_modes:
            self.text_by_mode[mode].append(
                text
            )


def _extract_mode_text(
    value: Any,
) -> dict[str, str]:
    if value is None:
        return {}

    parser = _ModeValueParser()

    try:
        parser.feed(
            html.unescape(
                str(value)
            )
        )

    except Exception:
        return {}

    output: dict[str, str] = {}

    for mode, parts in (
        parser.text_by_mode.items()
    ):
        text = normalize_whitespace(
            " ".join(parts)
        )

        if text:
            output[mode] = text

    return output


def _normalize_dashes(
    value: str,
) -> str:
    return (
        value
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
    )


def _extract_probability(
    value: Any,
) -> dict[str, Any] | None:
    """
    Convert one textual drop rate into a probability
    interval.

    Examples:
        0.67%       -> 0.0067
        1/560       -> 0.001785...
        4/5 (80%)   -> 0.8
        5-25%       -> [0.05, 0.25]
    """
    plain = strip_markup(value)

    if not plain:
        return None

    text = _normalize_dashes(
        plain.replace(",", "")
    )

    percent_range = re.search(
        r"(\d+(?:\.\d+)?)"
        r"\s*-\s*"
        r"(\d+(?:\.\d+)?)"
        r"\s*%",
        text,
    )

    if percent_range:
        minimum = (
            float(percent_range.group(1))
            / 100.0
        )

        maximum = (
            float(percent_range.group(2))
            / 100.0
        )

        return {
            "minimum": minimum,
            "maximum": maximum,
        }

    percent_match = re.search(
        r"(\d+(?:\.\d+)?)\s*%",
        text,
    )

    if percent_match:
        probability = (
            float(percent_match.group(1))
            / 100.0
        )

        return {
            "minimum": probability,
            "maximum": probability,
        }

    fraction_match = re.search(
        r"(\d+)\s*/\s*(\d+)",
        text,
    )

    if fraction_match:
        numerator = int(
            fraction_match.group(1)
        )

        denominator = int(
            fraction_match.group(2)
        )

        if denominator == 0:
            return None

        probability = (
            numerator / denominator
        )

        return {
            "minimum": probability,
            "maximum": probability,
            "odds": {
                "numerator": numerator,
                "denominator": denominator,
            },
        }

    return None


def _expand_all_mode_value(
    by_mode: dict[str, Any],
    enabled_modes: list[str],
) -> dict[str, Any]:
    """
    Expand an 'all' value into the explicitly enabled
    modes.
    """
    if (
        "all" not in by_mode
        or not enabled_modes
    ):
        return by_mode

    all_value = by_mode["all"]

    expanded = {
        mode: value
        for mode, value in by_mode.items()
        if mode != "all"
    }

    for mode in enabled_modes:
        expanded.setdefault(
            mode,
            all_value,
        )

    return expanded


def _propagate_expert_to_master(
    by_mode: dict[str, Any],
    enabled_modes: list[str],
) -> dict[str, Any]:
    """
    Expert-only loot normally remains available in
    Master Mode. Cargo explicitly marks those records
    with expert=1 and master=1.
    """
    if (
        "expert" in by_mode
        and "master" in enabled_modes
        and "master" not in by_mode
    ):
        by_mode["master"] = (
            by_mode["expert"]
        )

    return by_mode


def _parse_rate(
    value: Any,
    enabled_modes: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    raw_value = value

    if value is None or not str(value).strip():
        warnings.append(
            "missing_rate"
        )

        return {
            "by_mode": {},
            "per_item": False,
            "raw_value": raw_value,
            "plain_text": None,
        }

    plain_text = strip_markup(value)

    mode_text = _extract_mode_text(
        value
    )

    by_mode: dict[str, Any] = {}

    for mode, text in mode_text.items():
        parsed = _extract_probability(
            text
        )

        if parsed is not None:
            by_mode[mode] = parsed

    by_mode = _expand_all_mode_value(
        by_mode,
        enabled_modes,
    )

    by_mode = _propagate_expert_to_master(
        by_mode,
        enabled_modes,
    )

    if not by_mode:
        parsed = _extract_probability(
            value
        )

        if parsed is not None:
            target_modes = (
                enabled_modes
                if enabled_modes
                else ["all"]
            )

            by_mode = {
                mode: parsed
                for mode in target_modes
            }

    if not by_mode:
        warnings.append(
            "rate_not_parsed"
        )

    return {
        "by_mode": by_mode,

        "per_item": bool(
            plain_text
            and re.search(
                r"\beach\b",
                plain_text,
                flags=re.IGNORECASE,
            )
        ),

        "raw_value": raw_value,
        "plain_text": plain_text,
    }


def _extract_quantity(
    value: Any,
) -> dict[str, int] | None:
    plain = strip_markup(value)

    if not plain:
        return None

    text = _normalize_dashes(
        plain.replace(",", "")
    )

    text = text.lstrip(
        "~≈ "
    )

    range_match = re.search(
        r"(\d+)\s*-\s*(\d+)",
        text,
    )

    if range_match:
        return {
            "minimum": int(
                range_match.group(1)
            ),
            "maximum": int(
                range_match.group(2)
            ),
        }

    piece_match = re.search(
        r"\b(\d+)\s+pieces?\b",
        text,
        flags=re.IGNORECASE,
    )

    if piece_match:
        quantity = int(
            piece_match.group(1)
        )

        return {
            "minimum": quantity,
            "maximum": quantity,
        }

    integer_match = re.fullmatch(
        r"\s*(\d+)\s*",
        text,
    )

    if integer_match:
        quantity = int(
            integer_match.group(1)
        )

        return {
            "minimum": quantity,
            "maximum": quantity,
        }

    return None


def _extract_labeled_quantities(
    value: Any,
) -> dict[str, dict[str, int]]:
    """
    Parse quantity values such as:

        5-14 (Underground)
        3-10 (Cavern)
    """
    plain = strip_markup(value)

    if not plain:
        return {}

    text = _normalize_dashes(
        plain.replace(",", "")
    )

    matches = re.findall(
        r"([~≈]?\s*\d+"
        r"(?:\s*-\s*\d+)?)"
        r"\s*\(([^)]+)\)",
        text,
    )

    output: dict[
        str,
        dict[str, int]
    ] = {}

    for quantity_text, label_text in matches:
        quantity = _extract_quantity(
            quantity_text
        )

        label = _normalize_label(
            label_text
        )

        if quantity and label:
            output[label] = quantity

    return output


def _parse_quantity(
    value: Any,
    enabled_modes: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    """
    Parse quantity values.

    Empty Cargo quantity cells conventionally mean one
    item, so they are represented as an inferred
    default rather than treated as an error.
    """
    raw_value = value

    if value is None or not str(value).strip():
        default_quantity = {
            "minimum": 1,
            "maximum": 1,
        }

        target_modes = (
            enabled_modes
            if enabled_modes
            else ["all"]
        )

        return {
            "default": default_quantity,
            "by_mode": {
                mode: default_quantity
                for mode in target_modes
            },
            "by_condition": {},
            "inferred_default": True,
            "raw_value": raw_value,
            "plain_text": None,
        }

    plain_text = strip_markup(value)

    mode_text = _extract_mode_text(
        value
    )

    by_mode: dict[str, Any] = {}

    for mode, text in mode_text.items():
        parsed = _extract_quantity(
            text
        )

        if parsed is not None:
            by_mode[mode] = parsed

    by_mode = _expand_all_mode_value(
        by_mode,
        enabled_modes,
    )

    by_mode = _propagate_expert_to_master(
        by_mode,
        enabled_modes,
    )

    default_quantity = None

    if not by_mode:
        default_quantity = _extract_quantity(
            value
        )

        if default_quantity is not None:
            target_modes = (
                enabled_modes
                if enabled_modes
                else ["all"]
            )

            by_mode = {
                mode: default_quantity
                for mode in target_modes
            }

    by_condition = (
        _extract_labeled_quantities(
            value
        )
    )

    if (
        default_quantity is None
        and by_mode
    ):
        first_mode = next(
            iter(by_mode)
        )

        default_quantity = (
            by_mode[first_mode]
        )

    if (
        default_quantity is None
        and not by_condition
    ):
        warnings.append(
            "quantity_not_parsed"
        )

    return {
        "default": default_quantity,
        "by_mode": by_mode,
        "by_condition": by_condition,
        "inferred_default": False,
        "raw_value": raw_value,
        "plain_text": plain_text,
    }


def clean_drop_record(
    record: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise TypeError(
            "Drop record must be a dictionary."
        )

    data = record.get("data")

    if not isinstance(data, dict):
        raise ValueError(
            "Normalized drop record is missing "
            "its data dictionary."
        )

    warnings: list[str] = []

    item_name = (
        strip_markup(data.get("item"))
        or strip_markup(
            record.get("entity_name")
        )
    )

    source_name = (
        strip_markup(data.get("nameraw"))
        or strip_markup(data.get("_pageName"))
    )

    if not item_name:
        item_name = "Unknown Item"
        warnings.append(
            "missing_item_name"
        )

    if not source_name:
        source_name = "Unknown Source"
        warnings.append(
            "missing_source_name"
        )

    is_from_npc = parse_bool(
        data.get("isfromnpc"),
        empty_value=False,
    )

    source_id = parse_int(
        data.get("id")
        or record.get("entity_id")
    )

    enabled_modes = _enabled_modes(
        data
    )

    rate = _parse_rate(
        data.get("rate"),
        enabled_modes,
        warnings,
    )

    quantity = _parse_quantity(
        data.get("quantity"),
        enabled_modes,
        warnings,
    )

    custom_text = strip_markup(
        data.get("custom")
    )

    return {
        "source_catalog_id": record.get(
            "catalog_id"
        ),
        "source_row_id": data.get("_rowID"),
        "source_page": data.get("_pageName"),

        "record_type": "drop",

        "item": {
            "name": item_name,
            "normalized_name": (
                normalize_catalog_name(
                    item_name
                )
            ),
            "item_id": None,
        },

        "source": {
            "name": source_name,
            "normalized_name": (
                normalize_catalog_name(
                    source_name
                )
            ),
            "source_id": source_id,
            "source_type": (
                "npc"
                if is_from_npc
                else "container_or_other"
            ),
        },

        "availability": {
            mode: mode in enabled_modes
            for mode in _MODE_NAMES
        },

        "quantity": quantity,
        "chance": rate,

        "conditions": (
            [custom_text]
            if custom_text
            else []
        ),

        "raw_fields": {
            "quantity": data.get(
                "quantity"
            ),
            "rate": data.get("rate"),
            "custom": data.get("custom"),
            "isfromnpc": data.get(
                "isfromnpc"
            ),
            "normal": data.get("normal"),
            "expert": data.get("expert"),
            "master": data.get("master"),
            "source_id": data.get("id"),
        },

        "parse_status": (
            "ok"
            if not warnings
            else "partial"
        ),

        "parse_warnings": warnings,
    }


def clean_drops_file(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    report_path = Path(report_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Drops input file not found: "
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

    inferred_quantity_records = 0
    rate_parsed_records = 0
    quantity_parsed_records = 0

    source_type_counts: Counter[str] = (
        Counter()
    )

    warning_counts: Counter[str] = (
        Counter()
    )

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

            cleaned = clean_drop_record(
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

            if cleaned[
                "quantity"
            ]["inferred_default"]:
                inferred_quantity_records += 1

            if cleaned[
                "chance"
            ]["by_mode"]:
                rate_parsed_records += 1

            if (
                cleaned[
                    "quantity"
                ]["default"]
                is not None
                or cleaned[
                    "quantity"
                ]["by_condition"]
            ):
                quantity_parsed_records += 1

            source_type_counts.update(
                [
                    cleaned[
                        "source"
                    ]["source_type"]
                ]
            )

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
            "Drops input file contained "
            "no records."
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

        "rate_parsed_records": (
            rate_parsed_records
        ),

        "quantity_parsed_records": (
            quantity_parsed_records
        ),

        "inferred_quantity_records": (
            inferred_quantity_records
        ),

        "source_type_counts": dict(
            source_type_counts.most_common()
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
