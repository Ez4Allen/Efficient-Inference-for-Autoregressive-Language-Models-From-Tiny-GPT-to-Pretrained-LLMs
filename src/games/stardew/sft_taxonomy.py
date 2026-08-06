"""Versioned mapping from free-form SFT candidate categories to a controlled
reporting taxonomy (``task_family`` / ``topic``) and to the runtime intent
vocabulary consumed by :class:`~src.games.stardew.intent_router.StardewIntentRouter`.

This is the single place where category -> intent/task_family decisions are
made for the Stardew SFT candidate pool. Do not duplicate this mapping logic
in other scripts; import from here instead.

The candidate pool (as of dataset_version 1.0) contains ~490 free-form
category labels produced ad hoc during AI-assisted generation. Hand-mapping
every literal label is not maintainable, so classification is done with an
ordered list of substring rules evaluated against the lowercased category
string. The first matching rule wins. A category that matches nothing falls
back to ``task_family="other"`` / ``intent="unknown"`` and is reported as an
unmapped-category audit warning by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

TAXONOMY_VERSION = 1

TASK_FAMILIES = (
    "crop_planning",
    "fish_availability",
    "villager_gifts",
    "recipe_ingredients",
    "bundle_community_center",
    "item_acquisition",
    "guide_progression",
    "other",
)

# Intent values must stay a subset of the router's known vocabulary so this
# metadata remains compatible with StardewIntentRouter without inventing a
# parallel intent system.
ROUTER_INTENTS = (
    "entity",
    "crop_info",
    "crop_deadline",
    "fish_availability",
    "villager_info",
    "villager_gifts",
    "recipe",
    "recipes_using_item",
    "bundle",
    "bundles_requiring_item",
    "acquisition",
    "guide",
    "search",
    "unknown",
)

_UNMATCHED_TASK_FAMILY = "other"
_UNMATCHED_INTENT = "unknown"

# Ordered (task_family, intent, keywords) rules. Order matters: earlier rules
# take priority over later, broader ones (e.g. a bundle-donation category
# must be classified as bundle_community_center even though it also mentions
# "item", which would otherwise match the broad item_acquisition rule).
_RULES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "bundle_community_center",
        "bundle",
        (
            "bundle",
            "community_center",
            "community_upgrade",
            "joja",
            "field_office_donation",
            "grange_display",
            "museum_donation",
        ),
    ),
    (
        "villager_gifts",
        "villager_gifts",
        (
            "gift",
            "npc_",
            "villager",
            "birthday",
            "marriage",
            "spouse",
            "heart_event",
            "friendship",
            "horse_naming",
        ),
    ),
    (
        "fish_availability",
        "fish_availability",
        (
            "fish",
            "fishing",
            "tackle",
            "bait",
            "crab_pot",
            "spa_fishing",
        ),
    ),
    (
        "crop_planning",
        "crop_info",
        (
            "crop_",
            "seed_",
            "seeds_",
            "planting",
            "tree_",
            "fertilizer",
            "multi_seasonal_crop",
            "seasonal_growth",
            "seasonal_farming",
            "bush_",
            "giant_crop",
            "sprinkler",
        ),
    ),
    (
        "recipe_ingredients",
        "recipe",
        (
            "recipe",
            "cooking",
            "crafting",
            "tailoring",
            "dyeing",
            "forging",
            "sewing_machine",
            "tapper",
            "item_crafting",
        ),
    ),
    (
        "item_acquisition",
        "entity",
        (
            "acquisition",
            "purchasing",
            "purchase",
            "obtain",
            "furniture",
            "weapon",
            "hat_",
            "footwear",
            "ring_",
            "shop_",
            "shop",
            "_source",
            "sources",
            "_drop",
            "drops",
            "artisan_good",
            "book_",
            "catalogue",
            "staircase",
            "workbench",
            "chest_",
            "item_",
            "tool_",
        ),
    ),
    (
        "guide_progression",
        "guide",
        (
            "guide",
            "progression",
            "strategy",
            "quest",
            "festival",
            "perfection",
            "mastery",
            "achievement",
            "unlock",
            "special_order",
            "trivia",
            "lore",
            "secret",
            "event",
            "glitch",
            "bug",
            "cutscene",
            "soundtrack",
            "game_",
            "mechanics",
            "minigame",
            "combat",
            "enemy",
            "mine_",
            "mines_",
            "mining",
            "dungeon",
            "starting",
            "access",
            "location",
        ),
    ),
)


@dataclass(frozen=True)
class Classification:
    task_family: str
    topic: str
    intent: str
    matched: bool


def classify_category(category: str | None) -> Classification:
    """Classify a raw candidate ``category`` string.

    ``topic`` is the normalized (lowercase, trimmed) original category,
    preserved at fine granularity for reporting. ``task_family`` is the
    coarse bucket. ``matched`` is False when no rule applied, in which case
    the caller should emit an audit warning rather than guess further.
    """

    raw = (category or "").strip()
    topic = raw.lower()

    if not raw:
        return Classification(
            task_family=_UNMATCHED_TASK_FAMILY,
            topic="uncategorized",
            intent=_UNMATCHED_INTENT,
            matched=False,
        )

    for task_family, intent, keywords in _RULES:
        if any(keyword in topic for keyword in keywords):
            return Classification(
                task_family=task_family,
                topic=topic,
                intent=intent,
                matched=True,
            )

    return Classification(
        task_family=_UNMATCHED_TASK_FAMILY,
        topic=topic,
        intent=_UNMATCHED_INTENT,
        matched=False,
    )
