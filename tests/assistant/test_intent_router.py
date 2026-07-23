from __future__ import annotations

import pytest

from src.assistant import AssistantIntent, AssistantRequest, IntentRouter


@pytest.fixture()
def router() -> IntentRouter:
    return IntentRouter()


@pytest.mark.parametrize(
    ("question", "intent", "entity"),
    [
        ("How do I craft Night's Edge?", AssistantIntent.RECIPE, "Night's Edge"),
        ("夜之刃怎么合成？", AssistantIntent.RECIPE, "夜之刃"),
        ("What does Moon Lord drop?", AssistantIntent.DROPS_FROM_SOURCE, "Moon Lord"),
        ("Beam Sword 哪里掉？", AssistantIntent.DROPS_FOR_ITEM, "Beam Sword"),
        (
            "What can I craft with Terra Blade?",
            AssistantIntent.RECIPES_USING_ITEM,
            "Terra Blade",
        ),
        ("Moon Lord 属性是什么？", AssistantIntent.NPC, "Moon Lord"),
        ("What is Terra Blade?", AssistantIntent.SEARCH, "Terra Blade"),
    ],
)
def test_routes_common_questions(
    router: IntentRouter,
    question: str,
    intent: AssistantIntent,
    entity: str,
) -> None:
    decision = router.route(question)
    assert decision.intent == intent
    assert decision.entity == entity
    assert decision.confidence > 0.4


def test_extracts_mode_and_recipe_variant_preference(router: IntentRouter) -> None:
    decision = router.route(
        AssistantRequest(
            "In expert mode, what does Armored Skeleton drop?"
        )
    )
    assert decision.intent == AssistantIntent.DROPS_FROM_SOURCE
    assert decision.entity == "Armored Skeleton"
    assert decision.parameters["mode"] == "expert"

    legacy = router.route("Show all recipes for Night's Edge")
    assert legacy.intent == AssistantIntent.RECIPE
    assert legacy.entity == "Night's Edge"
    assert legacy.parameters["preferred_only"] is False


def test_extracts_disambiguation_identifiers(router: IntentRouter) -> None:
    item = router.route("item: Seaweed item id 753")
    assert item.intent == AssistantIntent.ITEM
    assert item.entity == "Seaweed"
    assert item.parameters["item_id"] == 753

    npc = router.route("npc: Armored Skeleton npc id 77")
    assert npc.intent == AssistantIntent.NPC
    assert npc.entity == "Armored Skeleton"
    assert npc.parameters["npc_id"] == 77
