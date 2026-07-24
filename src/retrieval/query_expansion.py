"""Deterministic bilingual query planning for guide retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass


ENGLISH_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "should",
    "the",
    "to",
    "what",
    "when",
    "where",
    "with",
}

# More specific Chinese phrases must appear before their shorter substrings.
# Python preserves insertion order, so this prevents a broad "Hardmode" hint from
# being the only semantic signal for an "immediately after entering Hardmode"
# question.
PHRASE_EXPANSIONS = {
    "进入困难模式后": [
        "early hardmode priorities",
        "after wall of flesh",
        "hardmode ores",
        "biome spread",
        "mechanical bosses preparation",
    ],
    "困难模式后": [
        "early hardmode priorities",
        "after wall of flesh",
        "hardmode ores",
        "biome spread",
        "mechanical bosses preparation",
    ],
    "肉山后": [
        "early hardmode priorities",
        "after wall of flesh",
        "hardmode ores",
        "biome spread",
        "mechanical bosses preparation",
    ],
    "第一晚": ["getting started", "first night", "early game", "shelter", "house building"],
    "第一天": ["getting started", "first day", "early game", "shelter"],
    "开局": ["getting started", "early game", "progression"],
    "新手": ["getting started", "beginner", "progression"],
    "困难模式": ["hardmode", "progression"],
    "机械三王": ["mechanical bosses", "destroyer", "twins", "skeletron prime"],
    "世纪之花": ["plantera", "strategy", "preparation"],
    "月亮领主": ["moon lord", "strategy"],
    "职业": ["class setups", "melee", "ranged", "magic", "summoner"],
    "配装": ["class setups", "equipment", "loadout"],
    "装备路线": ["armor progression", "weapon progression", "class setups"],
    "boss顺序": ["boss progression", "boss order", "game progression", "overview"],
    "boss流程": ["boss progression", "boss order", "game progression", "overview"],
    "流程": ["game progression", "walkthrough"],
    "攻略": ["guide", "strategy", "progression"],
    "竞技场": ["arena", "boss strategy", "preparation"],
    "场地": ["arena", "preparation", "building"],
    "腐化扩散": ["maintaining world purity", "corruption spread", "biome spread", "containment"],
    "猩红扩散": ["maintaining world purity", "crimson spread", "biome spread", "containment"],
    "环境扩散": ["maintaining world purity", "biome spread", "containment"],
    "房屋": ["house", "housing", "town npc"],
    "幸福度": ["npc happiness", "pylons"],
    "钓鱼": ["fishing", "fishing quests"],
    "微光": ["shimmer"],
    "专家模式": ["expert mode"],
    "大师模式": ["master mode"],
    "事件": ["events", "invasion", "strategies"],
}

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]*|[\u3400-\u9fff]+")


def _tokens(value: str) -> list[str]:
    return [
        token.casefold().strip("_-")
        for token in TOKEN_RE.findall(value)
        if token.casefold().strip("_-")
        and token.casefold().strip("_-") not in ENGLISH_STOPWORDS
        and not re.fullmatch(r"[\u3400-\u9fff]+", token)
    ]


def _normalize_title(value: str) -> str:
    value = value.casefold().removeprefix("guide:")
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


@dataclass(frozen=True)
class GuideQueryPlan:
    terms: tuple[str, ...]
    original_terms: tuple[str, ...]
    preferred_titles: tuple[str, ...] = ()
    preferred_sections: tuple[str, ...] = ()
    discouraged_sections: tuple[str, ...] = ()
    anchor_terms: tuple[str, ...] = ()
    profile: str = "general"
    advice_query: bool = True
    broad_scope: bool = False

    @property
    def normalized_preferred_titles(self) -> frozenset[str]:
        return frozenset(_normalize_title(value) for value in self.preferred_titles)


_QUERY_HINTS = (
    (
        ("first night", "first day", "getting started", "第一晚", "第一天", "开局", "新手"),
        ("Guide:Getting started", "Guide:Walkthrough", "Guide:Game progression"),
        ("The First Day", "Safety and House building", "First night", "Getting Started"),
    ),
    (
        (
            "entering hardmode",
            "enter hardmode",
            "after entering hardmode",
            "after wall of flesh",
            "hardmode start",
            "进入困难模式后",
            "困难模式后",
            "肉山后",
        ),
        ("Guide:Game progression", "Hardmode", "Guide:Walkthrough"),
        ("Early Hardmode", "Early Hardmode priorities", "Tips", "Hardmode"),
    ),
    (
        ("class setup", "class setups", "loadout", "职业", "配装", "装备路线"),
        ("Guide:Class setups", "Guide:Armor progression"),
        ("Pre-Hardmode", "Hardmode", "Class setups"),
    ),
    (
        ("biome spread", "world purity", "corruption spread", "crimson spread", "腐化扩散", "猩红扩散", "环境扩散"),
        ("Guide:Maintaining world purity", "Hardmode", "Biomes"),
        ("Biome spread", "Spread", "Containing and preventing biome spread", "Hardmode"),
    ),
    (
        ("boss progression", "boss order", "boss流程", "boss顺序"),
        ("Guide:Game progression", "Guide:Boss strategies", "Bosses"),
        ("Overview", "Boss progression", "Bosses", "Progression"),
    ),
    (
        ("housing", "house", "房屋"),
        ("House", "NPC happiness", "Town NPCs"),
        ("Requirements", "Housing", "House"),
    ),
    (
        ("fishing", "钓鱼"),
        ("Fishing",),
        ("Fishing", "Fishing quests"),
    ),
)


def _profile_for_query(text: str) -> tuple[str, tuple[str, ...], tuple[str, ...], bool]:
    lowered = re.sub(r"\s+", " ", text.casefold()).strip()
    compact_chinese = re.sub(r"\s+", "", text)

    if any(
        marker in lowered or marker in compact_chinese
        for marker in (
            "first night",
            "first day",
            "第一晚",
            "第一天",
        )
    ):
        return (
            "first_night",
            ("first", "night", "safety", "shelter", "house", "building"),
            ("Character Creation", "World Creation", "Exploring"),
            False,
        )

    if any(
        marker in lowered or marker in compact_chinese
        for marker in (
            "after entering hardmode",
            "entering hardmode",
            "after wall of flesh",
            "hardmode start",
            "进入困难模式后",
            "困难模式后",
            "肉山后",
        )
    ):
        return (
            "early_hardmode",
            (
                "early",
                "hardmode",
                "priorities",
                "wall",
                "flesh",
                "ores",
                "biome",
                "spread",
                "mechanical",
                "bosses",
                "preparation",
            ),
            (
                "Pre-Hardmode",
                "After Plantera",
                "Post-Plantera",
                "Hardmode Jungle",
                "The Frost Legion",
            ),
            False,
        )

    if any(
        marker in lowered or marker in compact_chinese
        for marker in (
            "boss progression",
            "boss order",
            "boss route",
            "recommended boss",
            "boss顺序",
            "boss流程",
        )
    ):
        return (
            "boss_progression",
            ("boss", "progression", "order", "overview"),
            ("After Plantera", "Mechanical bosses", "Late Pre-Hardmode"),
            True,
        )

    if any(
        marker in lowered or marker in compact_chinese
        for marker in (
            "biome spread",
            "world purity",
            "corruption spread",
            "crimson spread",
            "腐化扩散",
            "猩红扩散",
            "环境扩散",
        )
    ):
        return (
            "biome_spread",
            ("biome", "spread", "purity", "containment", "corruption", "crimson", "hallow"),
            ("Post-Plantera",),
            False,
        )

    return "general", (), (), False


def plan_guide_query(query: str) -> GuideQueryPlan:
    """Build retrieval terms plus title/section preferences for *query*."""

    text = str(query).strip()
    original_terms = _tokens(text)
    terms = list(original_terms)

    compact_chinese = re.sub(r"\s+", "", text)
    for phrase, expansions in PHRASE_EXPANSIONS.items():
        if phrase in compact_chinese:
            for expansion in expansions:
                terms.extend(_tokens(expansion))

    lowered = text.casefold()
    english_rules = (
        ("first night", ["getting", "started", "first", "night", "safety", "shelter", "house", "building"]),
        ("first day", ["getting", "started", "first", "day", "early", "game"]),
        (
            "entering hardmode",
            [
                "early",
                "hardmode",
                "priorities",
                "wall",
                "flesh",
                "ores",
                "biome",
                "spread",
                "mechanical",
                "bosses",
                "preparation",
            ],
        ),
        (
            "after entering hardmode",
            [
                "early",
                "hardmode",
                "priorities",
                "wall",
                "flesh",
                "ores",
                "biome",
                "spread",
                "mechanical",
                "bosses",
                "preparation",
            ],
        ),
        ("after wall of flesh", ["early", "hardmode", "priorities", "progression"]),
        ("prepare for", ["strategy", "preparation", "arena"]),
        ("class setup", ["class", "setups", "equipment", "loadout"]),
        ("biome spread", ["maintaining", "world", "purity", "spread", "containment"]),
        ("boss progression", ["boss", "game", "progression", "order", "overview"]),
        ("boss order", ["boss", "game", "progression", "order", "overview"]),
    )
    for phrase, additions in english_rules:
        if phrase in lowered:
            terms.extend(additions)

    preferred_titles: list[str] = []
    preferred_sections: list[str] = []
    compact_lowered = re.sub(r"\s+", " ", lowered).strip()
    for triggers, titles, sections in _QUERY_HINTS:
        if any(trigger.casefold() in compact_lowered or trigger in compact_chinese for trigger in triggers):
            preferred_titles.extend(titles)
            preferred_sections.extend(sections)

    profile, anchor_terms, discouraged_sections, broad_scope = _profile_for_query(text)
    terms.extend(anchor_terms)

    advice_markers = (
        "what should",
        "how should",
        "guide",
        "progression",
        "strategy",
        "prepare",
        "first night",
        "first day",
        "进入",
        "做什么",
        "怎么发展",
        "攻略",
        "流程",
        "准备",
    )
    advice_query = any(marker in lowered or marker in text for marker in advice_markers)

    return GuideQueryPlan(
        terms=tuple(dict.fromkeys(terms)),
        original_terms=tuple(dict.fromkeys(original_terms)),
        preferred_titles=tuple(dict.fromkeys(preferred_titles)),
        preferred_sections=tuple(dict.fromkeys(preferred_sections)),
        discouraged_sections=tuple(dict.fromkeys(discouraged_sections)),
        anchor_terms=tuple(dict.fromkeys(anchor_terms)),
        profile=profile,
        advice_query=advice_query,
        broad_scope=broad_scope,
    )


def expand_guide_query(query: str) -> list[str]:
    """Backward-compatible term-only query expansion."""

    return list(plan_guide_query(query).terms)
