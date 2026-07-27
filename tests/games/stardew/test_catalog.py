from __future__ import annotations

import sqlite3

from src.games.stardew import StardewFactService, StardewQueryStore


def test_database_integrity_and_counts(stardew_database):
    connection = sqlite3.connect(stardew_database)
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert connection.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 31
    connection.close()


def test_exact_and_chinese_alias_lookup(stardew_database):
    with StardewQueryStore(stardew_database) as store:
        result = store.get_crop("欧洲防风草")
        assert result["status"] == "found"
        assert result["match"]["name"] == "Parsnip"
        assert result["match"]["facts"]["growth_days"] == 4


def test_crop_deadline_calculation(stardew_database):
    with StardewFactService(stardew_database) as service:
        result = service.crop_deadline(
            "Parsnip",
            player_state={"season": "spring", "day": 24},
        )
    assert result["status"] == "found"
    assert result["facts"]["can_harvest_before_season_end"] is True
    assert result["facts"]["first_harvest_day"] == 28
    assert result["facts"]["latest_planting_day"] == 24


def test_crop_deadline_requires_state(stardew_database):
    with StardewFactService(stardew_database) as service:
        result = service.crop_deadline("Cauliflower")
    assert result["status"] == "needs_context"
    assert set(result["missing_context"]) == {"season", "day"}


def test_fish_availability_with_state(stardew_database):
    with StardewFactService(stardew_database) as service:
        result = service.fish_availability(
            "Catfish",
            player_state={
                "season": "spring",
                "weather": "rain",
                "time": "18:00",
                "location": "river",
            },
            require_current_state=True,
        )
    assert result["status"] == "found"
    assert result["facts"]["catchable_for_player_state"] is True
    assert len(result["facts"]["matching_windows"]) == 1


def test_bundle_reverse_lookup(stardew_database):
    with StardewFactService(stardew_database) as service:
        result = service.bundles_requiring_item("Catfish")
    assert result["status"] == "found"
    assert result["facts"]["bundles"] == ["River Fish Bundle"]


def test_post_midnight_fish_window(stardew_database):
    with StardewFactService(stardew_database) as service:
        result = service.fish_availability(
            "Bream",
            player_state={
                "season": "summer",
                "weather": "sunny",
                "time": "1:00am",
                "location": "river",
            },
            require_current_state=True,
        )
    assert result["facts"]["catchable_for_player_state"] is True


def test_all_day_window_does_not_shift_early_morning(stardew_database):
    with StardewFactService(stardew_database) as service:
        result = service.fish_availability(
            "Bullhead",
            player_state={
                "season": "spring",
                "weather": "sunny",
                "time": "1:00am",
                "location": "mountain_lake",
            },
            require_current_state=True,
        )
    assert result["facts"]["catchable_for_player_state"] is True


def test_storm_satisfies_rain_fishing_requirement(stardew_database):
    with StardewFactService(stardew_database) as service:
        result = service.fish_availability(
            "Catfish",
            player_state={
                "season": "spring",
                "weather": "storm",
                "time": "18:00",
                "location": "river",
            },
            require_current_state=True,
        )
    assert result["facts"]["catchable_for_player_state"] is True


def test_short_entity_name_does_not_match_inside_word(stardew_database):
    with StardewQueryStore(stardew_database) as store:
        assert store.resolve_entity_in_text("How should I feel about mining?") is None
        match = store.resolve_entity_in_text("Where can I catch Eel?")
    assert match is not None
    assert match["name"] == "Eel"


def test_bundle_mode_is_not_silently_mixed(stardew_database):
    with StardewFactService(stardew_database) as service:
        standard = service.bundle("River Fish Bundle", bundle_mode="standard")
        remixed = service.bundle("River Fish Bundle", bundle_mode="remixed")
    assert standard["status"] == "found"
    assert remixed["status"] == "not_found"


def test_record_validation_rejects_missing_provenance(stardew_database):
    import copy
    import json
    from pathlib import Path
    import pytest
    from src.games.stardew.database_builder import DEFAULT_FACTS_PATH, validate_record

    record = json.loads(Path(DEFAULT_FACTS_PATH).read_text(encoding="utf-8").splitlines()[0])
    invalid = copy.deepcopy(record)
    invalid["provenance"]["source_url"] = ""
    with pytest.raises(ValueError, match="source_url"):
        validate_record(invalid)
