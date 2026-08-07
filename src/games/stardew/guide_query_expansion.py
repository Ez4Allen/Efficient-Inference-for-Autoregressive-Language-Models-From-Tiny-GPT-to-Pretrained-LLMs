"""Bilingual lexical query expansion for Stardew Valley guides."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class StardewGuideQueryPlan:
    original_query: str
    terms: tuple[str, ...]
    profile: str
    preferred_titles: tuple[str, ...] = ()


EXPANSIONS: list[tuple[tuple[str, ...], str, tuple[str, ...], tuple[str, ...]]] = [
    (("first spring", "first year", "getting started", "开局", "第一年", "第一春", "春季优先"),
     "early_game", ("getting", "started", "first", "spring", "energy", "farming", "mines"),
     ("Getting Started", "Stardew Valley", "Seasons")),
    (("community center", "bundle", "bundles", "社区中心", "收集包", "献祭"),
     "community_center", ("community", "center", "bundle", "junimo", "pantry", "fish", "tank"),
     ("Bundles", "Community Center", "Remixed Bundles")),
    (("fish", "fishing", "catch", "钓鱼", "鱼", "哪里钓", "什么时候钓"),
     "fishing", ("fish", "fishing", "season", "weather", "time", "location"),
     ("Fishing", "Fishing Strategy", "Fish")),
    (("skull cavern", "沙漠矿洞", "骷髅洞穴"),
     "skull_cavern", ("skull", "cavern", "desert", "stairs", "bombs", "luck"),
     ("Skull Cavern",)),
    (("mine", "mines", "mining", "矿洞", "下矿", "采矿"),
     "mining", ("mines", "mining", "elevator", "combat", "ore"),
     ("The Mines", "Combat", "Skills")),
    (("cooking recipes", "unlock more cooking", "learn recipes", "烹饪配方", "解锁更多烹饪", "学习配方"),
     "cooking", ("cooking", "recipe", "television", "friendship", "skill", "shop"),
     ("Cooking", "Friendship", "Skills")),
    (("gift", "friendship", "love", "marriage", "礼物", "友谊", "好感", "结婚"),
     "relationships", ("friendship", "gifts", "loved", "villager", "marriage"),
     ("Friendship", "Villagers", "Marriage")),
    (("greenhouse", "温室"), "greenhouse", ("greenhouse", "crops", "fruit", "trees"), ("Greenhouse", "Crops")),
    (("ginger island", "姜岛"), "ginger_island", ("ginger", "island", "walnut", "volcano"), ("Ginger Island",)),
    (("crop", "crops", "profit", "plant", "作物", "种植", "收益"),
     "crops", ("crop", "growth", "season", "profit", "seed", "harvest"), ("Crops", "Farming", "Seasons")),
    (("profession", "professions", "skill", "技能", "职业"),
     "skills", ("skill", "profession", "level", "farming", "fishing", "mining"), ("Skills",)),
]


def _tokenize(value: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]*|[\u4e00-\u9fff]+", value.casefold())


def plan_stardew_query(query: str) -> StardewGuideQueryPlan:
    text = str(query).strip()
    folded = text.casefold()
    base_terms = _tokenize(text)
    profile = "general"
    additions: list[str] = []
    titles: tuple[str, ...] = ()
    for patterns, candidate_profile, terms, preferred in EXPANSIONS:
        if any(pattern in folded for pattern in patterns):
            profile = candidate_profile
            additions.extend(terms)
            titles = preferred
            break
    ordered: list[str] = []
    seen: set[str] = set()
    for term in [*base_terms, *additions]:
        normalized = term.strip().casefold()
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return StardewGuideQueryPlan(
        original_query=text,
        terms=tuple(ordered),
        profile=profile,
        preferred_titles=titles,
    )
