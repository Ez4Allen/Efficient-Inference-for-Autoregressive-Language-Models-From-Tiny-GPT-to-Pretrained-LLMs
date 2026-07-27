"""Normalization helpers for Stardew Valley names and player-state values."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

SEASONS = {"spring", "summer", "fall", "winter"}
WEATHER = {"sunny", "rain", "rain_totem", "any", "storm", "snow"}


def normalize_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value)).casefold()
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text)


def normalize_season(value: Any) -> str | None:
    if value is None:
        return None
    mapping = {
        "spring": "spring", "春": "spring", "春季": "spring",
        "summer": "summer", "夏": "summer", "夏季": "summer",
        "fall": "fall", "autumn": "fall", "秋": "fall", "秋季": "fall",
        "winter": "winter", "冬": "winter", "冬季": "winter",
    }
    normalized = mapping.get(str(value).strip().casefold())
    if normalized is None:
        raise ValueError(f"Unsupported Stardew season: {value!r}")
    return normalized


def normalize_weather(value: Any) -> str | None:
    if value is None:
        return None
    mapping = {
        "sunny": "sunny", "sun": "sunny", "clear": "sunny", "晴": "sunny", "晴天": "sunny",
        "rain": "rain", "rainy": "rain", "雨": "rain", "下雨": "rain", "雨天": "rain",
        "storm": "storm", "stormy": "storm", "雷雨": "storm",
        "snow": "snow", "snowy": "snow", "雪": "snow", "下雪": "snow",
        "rain_totem": "rain_totem", "rain totem": "rain_totem", "雨水图腾": "rain_totem",
        "any": "any", "任意": "any",
    }
    normalized = mapping.get(str(value).strip().casefold())
    if normalized is None:
        raise ValueError(f"Unsupported Stardew weather: {value!r}")
    return normalized


def parse_clock(value: Any) -> int | None:
    """Convert a clock value to minutes after midnight.

    Times after midnight may be represented using 24:00-26:00, which matches
    Stardew Valley's game-day convention.
    """

    if value is None:
        return None
    if isinstance(value, int):
        if value < 0:
            raise ValueError("Clock minutes cannot be negative.")
        return value
    text = str(value).strip().casefold().replace(" ", "")
    match = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?(am|pm)?", text)
    if not match:
        raise ValueError(f"Invalid Stardew time: {value!r}")
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    suffix = match.group(3)
    if minute >= 60:
        raise ValueError(f"Invalid Stardew time: {value!r}")
    if suffix:
        if hour < 1 or hour > 12:
            raise ValueError(f"Invalid Stardew time: {value!r}")
        if suffix == "am":
            hour = 0 if hour == 12 else hour
        else:
            hour = 12 if hour == 12 else hour + 12
    elif hour > 26:
        raise ValueError(f"Invalid Stardew time: {value!r}")
    return hour * 60 + minute


def format_clock(minutes: int | None) -> str | None:
    if minutes is None:
        return None
    hour, minute = divmod(int(minutes), 60)
    return f"{hour:02d}:{minute:02d}"
