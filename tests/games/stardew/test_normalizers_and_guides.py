from __future__ import annotations

import pytest

from src.games.stardew.guide_query_expansion import plan_stardew_query
from src.games.stardew.normalizers import (
    format_clock,
    normalize_season,
    normalize_weather,
    parse_clock,
)


def test_bilingual_season_normalization():
    assert normalize_season("春季") == "spring"
    assert normalize_season("Autumn") == "fall"


def test_bilingual_weather_normalization():
    assert normalize_weather("雨天") == "rain"
    assert normalize_weather("sunny") == "sunny"


def test_stardew_clock_supports_post_midnight_game_time():
    assert parse_clock("2am") == 120
    assert parse_clock("26:00") == 1560
    assert format_clock(1560) == "26:00"


def test_invalid_clock_is_rejected():
    with pytest.raises(ValueError):
        parse_clock("29:00")


def test_first_spring_query_expansion():
    plan = plan_stardew_query("第一年春天优先做什么？")
    assert plan.profile == "early_game"
    assert "spring" in plan.terms
    assert "Getting Started" in plan.preferred_titles


def test_community_center_query_expansion():
    plan = plan_stardew_query("How should I plan the Community Center bundles?")
    assert plan.profile == "community_center"
    assert "bundle" in plan.terms


def test_skull_cavern_query_precedes_generic_mine_expansion():
    plan = plan_stardew_query("沙漠矿洞应该怎么准备？")
    assert plan.profile == "skull_cavern"
    assert plan.preferred_titles == ("Skull Cavern",)


def test_cooking_progression_query_expansion():
    plan = plan_stardew_query("How do I unlock more cooking recipes?")
    assert plan.profile == "cooking"
    assert "Cooking" in plan.preferred_titles
