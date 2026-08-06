from __future__ import annotations

from src.games.stardew.sft_taxonomy import ROUTER_INTENTS, classify_category


def test_bundle_category_maps_to_bundle_family() -> None:
    result = classify_category("community_upgrade")
    assert result.task_family == "bundle_community_center"
    assert result.intent == "bundle"
    assert result.matched


def test_villager_gift_category_maps_to_villager_gifts() -> None:
    result = classify_category("npc_gifts")
    assert result.task_family == "villager_gifts"
    assert result.intent == "villager_gifts"
    assert result.matched


def test_fish_category_maps_to_fish_availability() -> None:
    result = classify_category("fishing_mechanics")
    assert result.task_family == "fish_availability"
    assert result.matched


def test_crop_category_maps_to_crop_planning() -> None:
    result = classify_category("crop_growth")
    assert result.task_family == "crop_planning"
    assert result.intent == "crop_info"


def test_recipe_category_maps_to_recipe_ingredients() -> None:
    result = classify_category("tailoring_dyeing")
    assert result.task_family == "recipe_ingredients"
    assert result.intent == "recipe"


def test_acquisition_category_maps_to_item_acquisition() -> None:
    result = classify_category("furniture_acquisition")
    assert result.task_family == "item_acquisition"


def test_unmapped_category_falls_back_to_other() -> None:
    result = classify_category("totally_unrecognized_topic_xyz")
    assert result.task_family == "other"
    assert result.intent == "unknown"
    assert result.matched is False


def test_empty_category_falls_back_to_other_without_matching() -> None:
    result = classify_category("")
    assert result.task_family == "other"
    assert result.matched is False
    assert result.topic == "uncategorized"

    result_none = classify_category(None)
    assert result_none.task_family == "other"
    assert result_none.matched is False


def test_topic_preserves_normalized_original_category() -> None:
    result = classify_category("Crop_Growth")
    assert result.topic == "crop_growth"


def test_bundle_rule_takes_priority_over_item_rule() -> None:
    # "community_upgrade" mentions nothing acquisition-like, but a category
    # combining both signals should still resolve to the higher-priority
    # bundle family per the ordered rule list.
    result = classify_category("community_upgrade_item")
    assert result.task_family == "bundle_community_center"


def test_all_classification_intents_are_router_compatible() -> None:
    sample_categories = [
        "community_upgrade", "npc_gifts", "fishing_mechanics", "crop_growth",
        "tailoring_dyeing", "furniture_acquisition", "guide_progression",
        "unrecognized_xyz",
    ]
    for category in sample_categories:
        result = classify_category(category)
        assert result.intent in ROUTER_INTENTS
