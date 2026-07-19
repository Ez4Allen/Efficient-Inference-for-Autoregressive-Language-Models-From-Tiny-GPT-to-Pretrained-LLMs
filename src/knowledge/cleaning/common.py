
from __future__ import annotations

import html
import re

from fractions import Fraction
from html.parser import HTMLParser
from typing import Any


_EMPTY_VALUES = {
    "",
    "-",
    "—",
    "–",
    "none",
    "null",
    "n/a",
    "na",
}


def normalize_whitespace(
    value: str,
) -> str:
    """
    Collapse repeated whitespace and remove leading
    and trailing spaces.
    """
    if not isinstance(value, str):
        raise TypeError(
            "normalize_whitespace expects a string."
        )

    return re.sub(
        r"\s+",
        " ",
        value,
    ).strip()


def _is_empty_text(
    value: Any,
) -> bool:
    if value is None:
        return True

    if not isinstance(value, str):
        return False

    return (
        normalize_whitespace(value).casefold()
        in _EMPTY_VALUES
    )


def strip_markup(
    value: Any,
) -> str | None:
    """
    Convert common Terraria Wiki HTML and MediaWiki
    markup into readable plain text.

    Examples:
        [[Moon Lord|Moon Lord's Core]]
        -> Moon Lord's Core

        <span>45000</span>
        -> 45000

        A^B^C is not split here. Use split_caret_list().
    """
    if value is None:
        return None

    text = str(value)

    if not text.strip():
        return None

    text = html.unescape(text)

    # Remove category annotations.
    text = re.sub(
        r"\[\[Category:[^\]]+\]\]",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove file/image links.
    text = re.sub(
        r"\[\[(?:File|Image):[^\]]+\]\]",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # [[Target|Visible text]] -> Visible text
    text = re.sub(
        r"\[\[[^|\]]+\|([^\]]+)\]\]",
        r"\1",
        text,
    )

    # [[Visible text]] -> Visible text
    text = re.sub(
        r"\[\[([^\]]+)\]\]",
        r"\1",
        text,
    )

    # Remove Terraria annotation fragments such as:
    # #i:old, #n:note
    text = re.sub(
        r"#(?:i|n):[^#\s]+",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Convert common HTML separators before removing tags.
    text = re.sub(
        r"<\s*br\s*/?\s*>",
        " / ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"</\s*(?:p|div|li|tr)\s*>",
        " / ",
        text,
        flags=re.IGNORECASE,
    )

    # Remove remaining HTML tags.
    text = re.sub(
        r"<[^>]+>",
        "",
        text,
    )

    text = html.unescape(text)

    # Remove leftover external-link wrappers:
    # [https://example.com label] -> label
    text = re.sub(
        r"\[https?://\S+\s+([^\]]+)\]",
        r"\1",
        text,
    )

    text = normalize_whitespace(text)

    text = re.sub(
        r"(?:\s*/\s*){2,}",
        " / ",
        text,
    )

    text = text.strip(" /")

    return text or None


def parse_bool(
    value: Any,
    *,
    empty_value: bool | None = None,
) -> bool | None:
    """
    Parse common Cargo boolean representations.

    Accepted true values:
        1, true, yes, y

    Accepted false values:
        0, false, no, n

    Empty strings return empty_value.
    """
    if value is None:
        return empty_value

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        if value == 1:
            return True

        if value == 0:
            return False

    text = normalize_whitespace(
        str(value)
    ).casefold()

    if text in _EMPTY_VALUES:
        return empty_value

    if text in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }:
        return True

    if text in {
        "0",
        "false",
        "no",
        "n",
        "off",
    }:
        return False

    return None


def parse_number(
    value: Any,
) -> int | float | None:
    """
    Parse a plain integer, decimal or fraction.

    Examples:
        "1,250" -> 1250
        "3.5"    -> 3.5
        "1/4"    -> 0.25
    """
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return value

    text = strip_markup(value)

    if text is None:
        return None

    text = text.replace(",", "").strip()

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

    if re.fullmatch(
        r"[+-]?\d+\s*/\s*\d+",
        text,
    ):
        numerator, denominator = re.split(
            r"\s*/\s*",
            text,
        )

        denominator_value = int(denominator)

        if denominator_value == 0:
            return None

        return float(
            Fraction(
                int(numerator),
                denominator_value,
            )
        )

    return None


def parse_int(
    value: Any,
) -> int | None:
    """
    Parse a value only when it represents an integer.
    """
    number = parse_number(value)

    if number is None:
        return None

    if isinstance(number, int):
        return number

    if float(number).is_integer():
        return int(number)

    return None


def parse_float(
    value: Any,
) -> float | None:
    number = parse_number(value)

    if number is None:
        return None

    return float(number)


def parse_percent(
    value: Any,
) -> float | None:
    """
    Parse a percentage or fraction into a probability.

    Examples:
        "0.67%" -> 0.0067
        "80%"   -> 0.8
        "1/560" -> 0.001785714...
    """
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    text = strip_markup(value)

    if text is None:
        return None

    text = text.replace(",", "")

    percent_match = re.fullmatch(
        r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*%\s*",
        text,
    )

    if percent_match:
        return (
            float(percent_match.group(1))
            / 100.0
        )

    fraction_match = re.fullmatch(
        r"\s*(\d+)\s*/\s*(\d+)\s*",
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

        return numerator / denominator

    return None


def parse_numeric_range(
    value: Any,
) -> dict[str, int | float] | None:
    """
    Parse a single number or numeric range.

    Examples:
        "1"    -> {"minimum": 1, "maximum": 1}
        "2-5"  -> {"minimum": 2, "maximum": 5}
        "2–5"  -> {"minimum": 2, "maximum": 5}
        "~2–4" -> {"minimum": 2, "maximum": 4}
    """
    if value is None:
        return None

    text = strip_markup(value)

    if text is None:
        return None

    text = text.replace(",", "")
    text = text.replace("−", "-")
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.lstrip("~≈ ")

    single_number = parse_number(text)

    if single_number is not None:
        return {
            "minimum": single_number,
            "maximum": single_number,
        }

    range_match = re.fullmatch(
        r"\s*"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))"
        r"\s*-\s*"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))"
        r"\s*",
        text,
    )

    if not range_match:
        return None

    minimum = parse_number(
        range_match.group(1)
    )

    maximum = parse_number(
        range_match.group(2)
    )

    if (
        minimum is None
        or maximum is None
    ):
        return None

    return {
        "minimum": minimum,
        "maximum": maximum,
    }


def split_caret_list(
    value: Any,
) -> list[str]:
    """
    Split Cargo list fields that use '^' as the
    separator.
    """
    if value is None:
        return []

    if isinstance(value, list):
        raw_values = value

    else:
        raw_values = str(value).split("^")

    output: list[str] = []
    seen: set[str] = set()

    for raw_value in raw_values:
        cleaned_value = strip_markup(
            raw_value
        )

        if not cleaned_value:
            continue

        if cleaned_value in seen:
            continue

        seen.add(cleaned_value)
        output.append(cleaned_value)

    return output


def parse_coin_value(
    value: Any,
) -> int | None:
    """
    Convert Terraria coin markup into total copper.

    Priority:
    1. data-sort-value attribute;
    2. textual PC / GC / SC / CC representation.
    """
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value

    text = str(value)

    sort_value_match = re.search(
        r'data-sort-value\s*=\s*["\'](\d+)["\']',
        text,
        flags=re.IGNORECASE,
    )

    if sort_value_match:
        return int(
            sort_value_match.group(1)
        )

    plain_text = strip_markup(text)

    if not plain_text:
        return None

    total_copper = 0
    matched = False

    coin_multipliers = {
        "pc": 1_000_000,
        "platinum": 1_000_000,

        "gc": 10_000,
        "gold": 10_000,

        "sc": 100,
        "silver": 100,

        "cc": 1,
        "copper": 1,
    }

    for amount_text, unit_text in re.findall(
        r"(\d+)\s*"
        r"(PC|GC|SC|CC|"
        r"Platinum|Gold|Silver|Copper)"
        r"(?:\s+Coins?)?",
        plain_text,
        flags=re.IGNORECASE,
    ):
        multiplier = coin_multipliers[
            unit_text.casefold()
        ]

        total_copper += (
            int(amount_text)
            * multiplier
        )

        matched = True

    return total_copper if matched else None


class _ModeTextParser(HTMLParser):
    """
    Extract text while tracking Terraria mode-specific
    CSS classes.
    """

    def __init__(self) -> None:
        super().__init__()

        self.mode_stack: list[set[str]] = []
        self.text_by_mode: dict[str, list[str]] = {
            "all": [],
            "normal": [],
            "expert": [],
            "master": [],
        }

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

        own_modes: set[str] = set()

        if "m-all" in class_tokens:
            own_modes.add("all")

        if (
            "m-normal" in class_tokens
            or "m-journey" in class_tokens
        ):
            own_modes.add("normal")

        if "m-expert-master" in class_tokens:
            own_modes.update(
                {
                    "expert",
                    "master",
                }
            )

        if (
            "m-expert" in class_tokens
            or (
                "expert" in class_tokens
                and "mode-exclusive" in class_tokens
            )
        ):
            own_modes.add("expert")

        if (
            "m-master" in class_tokens
            or (
                "master" in class_tokens
                and "mode-exclusive" in class_tokens
            )
        ):
            own_modes.add("master")

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

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[
            tuple[str, str | None]
        ],
    ) -> None:
        # Self-closing tags do not contribute useful text.
        return

    def handle_endtag(
        self,
        tag: str,
    ) -> None:
        if self.mode_stack:
            self.mode_stack.pop()

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


def parse_mode_numbers(
    value: Any,
) -> dict[str, list[int | float]]:
    """
    Extract numeric values grouped by Terraria mode.

    This deliberately returns lists because fields such
    as damage may contain several attacks.

    Example result:
        {
            "normal": [150, 60, 140],
            "expert": [300, 120, 280],
            "master": [450, 180, 420],
        }
    """
    if value is None:
        return {}

    raw_text = str(value)

    # Do not run the plain-number fallback on fields
    # containing Terraria mode markup. Removing those
    # tags can concatenate several mode values into one
    # incorrect number.
    has_mode_markup = bool(
        re.search(
            r'class\s*=\s*["\'][^"\']*'
            r'\b(?:'
            r'm-all|'
            r'm-normal|'
            r'm-journey|'
            r'm-expert|'
            r'm-expert-master|'
            r'm-master|'
            r'mode-content'
            r')\b',
            raw_text,
            flags=re.IGNORECASE,
        )
    )

    if not has_mode_markup:
        plain_text = strip_markup(
            value
        )

        if plain_text is not None:
            numeric_text = (
                plain_text
                .replace(",", "")
                .strip()
            )

            if re.fullmatch(
                r"[+-]?"
                r"(?:\d+(?:\.\d*)?|\.\d+)",
                numeric_text,
            ):
                plain_number = parse_number(
                    numeric_text
                )

                if plain_number is not None:
                    return {
                        "all": [plain_number],
                    }

    text = raw_text

    parser = _ModeTextParser()

    try:
        parser.feed(
            html.unescape(text)
        )

    except Exception:
        return {}

    result: dict[
        str,
        list[int | float]
    ] = {}

    for mode, text_parts in (
        parser.text_by_mode.items()
    ):
        values: list[int | float] = []

        for text_part in text_parts:
            for number_text in re.findall(
                r"(?<![\w.])"
                r"[+-]?"
                r"(?:\d{1,3}(?:,\d{3})+|\d+)"
                r"(?:\.\d+)?"
                r"(?![\w.])",
                text_part,
            ):
                number = parse_number(
                    number_text
                )

                if number is None:
                    continue

                values.append(number)

        if values:
            result[mode] = values

    return result


def _normalize_progression_label(
    value: str | None,
) -> str:
    """
    Convert a progression title into a stable key.

    Examples:
        Pre-Hardmode  -> pre_hardmode
        Hardmode      -> hardmode
        Post-Plantera -> post_plantera
    """
    if value is None:
        return "default"

    value = html.unescape(value)
    value = normalize_whitespace(value)

    if not value:
        return "default"

    normalized = re.sub(
        r"[^a-z0-9]+",
        "_",
        value.casefold(),
    ).strip("_")

    return normalized or "default"


class _ModeCoinParser(HTMLParser):
    """
    Parse Terraria coin values while tracking both
    difficulty mode and progression-stage labels.
    """

    def __init__(self) -> None:
        super().__init__()

        self.context_stack: list[
            tuple[set[str], str]
        ] = []

        self.values: dict[
            str,
            dict[str, list[int]]
        ] = {}

    @staticmethod
    def _extract_modes(
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

        if (
            "m-expert" in class_tokens
            or (
                "expert" in class_tokens
                and "mode-exclusive" in class_tokens
                and "m-expert-master"
                not in class_tokens
            )
        ):
            modes.add("expert")

        if (
            "m-master" in class_tokens
            or (
                "master" in class_tokens
                and "mode-exclusive" in class_tokens
            )
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

        inherited_modes: set[str] = set()
        inherited_stage = "default"

        if self.context_stack:
            inherited_modes = set(
                self.context_stack[-1][0]
            )

            inherited_stage = (
                self.context_stack[-1][1]
            )

        own_modes = self._extract_modes(
            class_tokens
        )

        active_modes = (
            own_modes
            if own_modes
            else inherited_modes
        )

        active_stage = inherited_stage

        # Wiki uses spans such as:
        # <span class="s" title="Pre-Hardmode">
        if (
            tag.casefold() == "span"
            and "s" in class_tokens
            and "coin" not in class_tokens
        ):
            title = attribute_map.get("title")

            if title:
                active_stage = (
                    _normalize_progression_label(
                        title
                    )
                )

        self.context_stack.append(
            (
                set(active_modes),
                active_stage,
            )
        )

        if (
            tag.casefold() == "span"
            and "coin" in class_tokens
        ):
            sort_value = attribute_map.get(
                "data-sort-value"
            )

            if (
                sort_value
                and re.fullmatch(
                    r"\d+",
                    sort_value,
                )
            ):
                copper_value = int(sort_value)

                target_modes = (
                    active_modes
                    if active_modes
                    else {"all"}
                )

                for mode in target_modes:
                    stage_values = (
                        self.values.setdefault(
                            mode,
                            {},
                        )
                    )

                    stage_values.setdefault(
                        active_stage,
                        [],
                    ).append(copper_value)

    def handle_endtag(
        self,
        tag: str,
    ) -> None:
        if self.context_stack:
            self.context_stack.pop()


def parse_mode_coin_values(
    value: Any,
) -> dict[
    str,
    dict[str, int | list[int]]
]:
    """
    Parse Terraria coin HTML into mode/stage values.

    Example:

        {
            "normal": {
                "default": 60
            },
            "expert": {
                "pre_hardmode": 150,
                "hardmode": 240,
                "post_plantera": 360
            },
            "master": {
                "pre_hardmode": 150,
                "hardmode": 240,
                "post_plantera": 360
            }
        }
    """
    if value is None or not str(value).strip():
        return {}

    parser = _ModeCoinParser()

    try:
        parser.feed(
            html.unescape(
                str(value)
            )
        )

    except Exception:
        return {}

    output: dict[
        str,
        dict[str, int | list[int]]
    ] = {}

    for mode, stage_values in (
        parser.values.items()
    ):
        output[mode] = {}

        for stage, values in (
            stage_values.items()
        ):
            unique_values: list[int] = []

            for coin_value in values:
                if coin_value not in unique_values:
                    unique_values.append(
                        coin_value
                    )

            if len(unique_values) == 1:
                output[mode][stage] = (
                    unique_values[0]
                )

            else:
                output[mode][stage] = (
                    unique_values
                )

    # Support simple non-HTML coin representations.
    if not output:
        fallback_value = parse_coin_value(
            value
        )

        # Some Cargo rows store a plain integer in the
        # money field. Cargo's numeric value represents
        # total copper.
        if fallback_value is None:
            fallback_value = parse_int(
                value
            )

        if fallback_value is not None:
            output = {
                "all": {
                    "default": fallback_value,
                }
            }

    return output


def parse_labeled_numbers(
    value: Any,
) -> dict[str, int | float]:
    """
    Parse values with explicit labels.

    Examples:
        14 (T1) / 18 (T2) / 20 (T3)

        -> {
            "t1": 14,
            "t2": 18,
            "t3": 20,
        }
    """
    text = strip_markup(value)

    if not text:
        return {}

    matches = re.findall(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))"
        r"\s*\(\s*([^)]+?)\s*\)",
        text,
    )

    output: dict[str, int | float] = {}

    for number_text, label_text in matches:
        number = parse_number(
            number_text
        )

        if number is None:
            continue

        label = re.sub(
            r"[^a-z0-9]+",
            "_",
            label_text.casefold(),
        ).strip("_")

        if not label:
            continue

        output[label] = number

    return output

