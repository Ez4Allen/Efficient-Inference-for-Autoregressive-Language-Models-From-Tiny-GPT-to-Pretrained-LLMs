from __future__ import annotations

from src.games.stardew import StardewAssistant


def test_bilingual_villager_gift_answer(stardew_database):
    assistant = StardewAssistant(database_path=stardew_database, auto_build=False)
    try:
        result = assistant.answer("阿比盖尔喜欢什么礼物？")
    finally:
        assistant.close()
    assert result.status == "found"
    assert result.intent == "villager_gifts"
    assert result.entity == "Abigail"
    assert "Amethyst" in result.answer
    assert "[S1]" in result.answer


def test_crop_deadline_route_needs_context(stardew_database):
    assistant = StardewAssistant(database_path=stardew_database, auto_build=False)
    try:
        result = assistant.answer("Can I still plant Cauliflower in time?")
    finally:
        assistant.close()
    assert result.status == "needs_context"
    assert set(result.context_payload["player_state"].keys()) >= {"season", "day"}


def test_false_premise_refusal(stardew_database):
    assistant = StardewAssistant(database_path=stardew_database, auto_build=False)
    try:
        result = assistant.answer("How long does Void Melon take to grow?")
    finally:
        assistant.close()
    assert result.status == "not_found"
    assert "不会猜测" in result.answer or "will not guess" in result.answer


def test_crop_deadline_detects_if_planted_wording(stardew_database):
    assistant = StardewAssistant(database_path=stardew_database, auto_build=False)
    try:
        result = assistant.answer("Can I harvest Yam if planted on Fall day 18?")
    finally:
        assistant.close()
    assert result.intent == "crop_deadline"
    assert result.status == "found"
    assert result.facts["first_harvest_day"] == 28


def test_false_premise_fish_name_does_not_resolve_by_suffix(stardew_database):
    assistant = StardewAssistant(database_path=stardew_database, auto_build=False)
    try:
        result = assistant.answer("Where can I catch Galaxy Catfish?")
    finally:
        assistant.close()
    assert result.intent == "fish_availability"
    assert result.status == "not_found"
    assert result.entity == "Galaxy Catfish"


def test_chinese_crop_deadline_wording(stardew_database):
    assistant = StardewAssistant(database_path=stardew_database, auto_build=False)
    try:
        result = assistant.answer("秋季第15天种Pumpkin还能收获吗？")
    finally:
        assistant.close()
    assert result.intent == "crop_deadline"
    assert result.status == "found"
    assert result.facts["first_harvest_day"] == 28


def test_fish_condition_wording_routes_to_fish_availability(stardew_database):
    assistant = StardewAssistant(database_path=stardew_database, auto_build=False)
    try:
        result = assistant.answer("What conditions are required for Pufferfish?")
    finally:
        assistant.close()
    assert result.intent == "fish_availability"
    assert result.status == "found"
    assert result.entity == "Pufferfish"


def test_acquisition_answer_has_structured_sources(stardew_database):
    assistant = StardewAssistant(database_path=stardew_database, auto_build=False)
    try:
        result = assistant.answer("Where can I buy Return Scepter?")
    finally:
        assistant.close()
    assert result.intent == "acquisition"
    assert result.status == "found"
    assert any(source["source_name"] == "Krobus" for source in result.facts["sources"])
    assert "2000000" in result.answer


def test_remixed_bundle_is_reported_as_partial(stardew_database):
    assistant = StardewAssistant(database_path=stardew_database, auto_build=False)
    try:
        result = assistant.answer(
            "What is in the remixed River Fish Bundle?",
            player_state={"bundle_mode": "remixed"},
        )
    finally:
        assistant.close()
    assert result.intent == "bundle"
    assert result.status == "partial"
    assert "will not present Standard Bundle data as Remixed" in result.answer
