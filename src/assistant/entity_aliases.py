"""Small bilingual name-alias layer for common Terraria entities.

The catalog remains the source of factual data.  These aliases only map user
surface forms to canonical English catalog names; they do not add game facts.
"""

from __future__ import annotations

from src.knowledge.catalog_store import normalize_catalog_name


_CANONICAL_ALIASES = {
    "夜之刃": "Night's Edge",
    "泰拉之刃": "Terra Blade",
    "天顶剑": "Zenith",
    "月亮领主": "Moon Lord",
    "月总": "Moon Lord",
    "机械魔眼": "Mechanical Eye",
    "双子魔眼": "The Twins",
    "克苏鲁之眼": "Eye of Cthulhu",
    "世界吞噬者": "Eater of Worlds",
    "血肉墙": "Wall of Flesh",
    "肉山": "Wall of Flesh",
    "世纪之花": "Plantera",
    "石巨人": "Golem",
    "骷髅王": "Skeletron",
    "机械骷髅王": "Skeletron Prime",
    "毁灭者": "The Destroyer",
    "光之女皇": "Empress of Light",
    "猪龙鱼公爵": "Duke Fishron",
    "史莱姆王": "King Slime",
    "光束剑": "Beam Sword",
    "装甲骷髅": "Armored Skeleton",
    "海草": "Seaweed",
    "暗影鳞片": "Shadow Scale",
}

ALIASES = {
    normalize_catalog_name(alias): canonical
    for alias, canonical in _CANONICAL_ALIASES.items()
}


def resolve_entity_alias(value: str) -> str:
    """Return the canonical catalog name for a known surface alias."""

    normalized = normalize_catalog_name(value)
    return ALIASES.get(normalized, value)
