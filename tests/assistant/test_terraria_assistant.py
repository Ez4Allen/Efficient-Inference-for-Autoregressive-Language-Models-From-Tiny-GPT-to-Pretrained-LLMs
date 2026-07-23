from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.assistant import AssistantIntent, TerrariaAssistant


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_english_recipe_question(assistant_catalog) -> None:
    with TerrariaAssistant(assistant_catalog.database_path) as assistant:
        response = assistant.answer("How do I craft Night's Edge?")

    assert response.status == "found"
    assert response.intent == AssistantIntent.RECIPE
    assert response.entity == "Night's Edge"
    assert response.facts["variant_count"] == 2
    assert "Volcano ×1" in response.answer
    assert "Demon Altar" in response.answer
    assert "Fiery Greatsword" not in response.answer
    assert any(row["entity_type"] == "recipe" for row in response.evidence)


def test_chinese_recipe_alias_and_rendering(assistant_catalog) -> None:
    with TerrariaAssistant(assistant_catalog.database_path) as assistant:
        response = assistant.answer("夜之刃怎么合成？")

    assert response.status == "found"
    assert response.intent == AssistantIntent.RECIPE
    assert response.entity == "Night's Edge"
    assert "Night's Edge" in response.answer
    assert "材料" in response.answer
    assert "制作站" in response.answer


def test_item_drop_source_with_mode_in_question(assistant_catalog) -> None:
    with TerrariaAssistant(assistant_catalog.database_path) as assistant:
        response = assistant.answer(
            "Where can I get Beam Sword in expert mode?"
        )

    assert response.status == "found"
    assert response.intent == AssistantIntent.DROPS_FOR_ITEM
    assert response.route.parameters["mode"] == "expert"
    assert "Armored Skeleton" in response.answer
    assert "0.67%" in response.answer


def test_source_drop_question_in_chinese(assistant_catalog) -> None:
    with TerrariaAssistant(assistant_catalog.database_path) as assistant:
        response = assistant.answer("装甲骷髅掉什么？", mode="expert")

    assert response.status == "found"
    assert response.intent == AssistantIntent.DROPS_FROM_SOURCE
    assert response.entity == "Armored Skeleton"
    assert "Armor Polish" in response.answer
    assert "1.99%" in response.answer
    assert "Beam Sword" in response.answer


def test_reverse_recipe_query(assistant_catalog) -> None:
    with TerrariaAssistant(assistant_catalog.database_path) as assistant:
        response = assistant.answer("泰拉之刃能合成什么？")

    assert response.status == "found"
    assert response.intent == AssistantIntent.RECIPES_USING_ITEM
    assert response.entity == "Terra Blade"
    assert "Zenith" in response.answer


def test_broad_lookup_resolves_to_item_facts(assistant_catalog) -> None:
    with TerrariaAssistant(assistant_catalog.database_path) as assistant:
        response = assistant.answer("What is Terra Blade?")

    assert response.status == "found"
    assert response.intent == AssistantIntent.ITEM
    assert response.facts["item_id"] == 757
    assert "damage 85" in response.answer
    assert "Sell value" in response.answer
    assert "catalog_type_resolution" in response.route.reason_codes


def test_unique_npc_stats_and_chinese_alias(assistant_catalog) -> None:
    with TerrariaAssistant(assistant_catalog.database_path) as assistant:
        response = assistant.answer("月亮领主属性是什么？", mode="master")

    assert response.status == "found"
    assert response.intent == AssistantIntent.NPC
    assert response.entity == "Moon Lord"
    assert response.facts["npc_id"] == 396
    assert "大师模式属性" in response.answer


def test_ambiguous_item_requires_clarification(assistant_catalog) -> None:
    with TerrariaAssistant(assistant_catalog.database_path) as assistant:
        response = assistant.answer("item: Seaweed")

    assert response.status == "clarification"
    assert response.intent == AssistantIntent.ITEM
    assert len(response.candidates) == 2
    assert {row["item_id"] for row in response.candidates} == {753, 2338}
    assert "multiple" in response.answer.casefold()


def test_item_id_disambiguates_ambiguous_item(assistant_catalog) -> None:
    with TerrariaAssistant(assistant_catalog.database_path) as assistant:
        response = assistant.answer("item: Seaweed item id 753")

    assert response.status == "found"
    assert response.facts["item_id"] == 753
    assert response.facts["internal_name"] == "Seaweed"


def test_npc_family_requires_clarification(assistant_catalog) -> None:
    with TerrariaAssistant(assistant_catalog.database_path) as assistant:
        response = assistant.answer("What are Armored Skeleton stats?")

    assert response.status == "clarification"
    assert response.intent == AssistantIntent.NPC
    assert {row["npc_id"] for row in response.candidates} == {-15, 77}


def test_grounded_not_found_response(assistant_catalog) -> None:
    with TerrariaAssistant(assistant_catalog.database_path) as assistant:
        response = assistant.answer("How do I craft Void Slime King?")

    assert response.status == "not_found"
    assert "will not invent" in response.answer
    assert response.facts is None
    assert response.evidence == []


def test_context_bundle_contains_fact_service_evidence(assistant_catalog) -> None:
    with TerrariaAssistant(assistant_catalog.database_path) as assistant:
        response = assistant.answer(
            "How do I craft Night's Edge?",
            include_debug=True,
        )
        context_payload = json.loads(
            assistant.context_json("How do I craft Night's Edge?")
        )

    assert response.context is not None
    assert response.context.payload["status"] == "found"
    assert response.context.payload["facts"]["result_name"] == "Night's Edge"
    assert "Answer only from the Terraria evidence" in response.context.text
    assert response.debug["timings_seconds"]["total"] >= 0
    assert context_payload["entity"] == "Night's Edge"


def test_all_recipe_variants_can_be_requested(assistant_catalog) -> None:
    with TerrariaAssistant(assistant_catalog.database_path) as assistant:
        response = assistant.answer(
            "How do I craft Night's Edge?",
            preferred_only=False,
        )

    assert response.status == "found"
    assert response.facts["variant_count"] == 4
    assert "Fiery Greatsword" in response.answer


def test_cli_one_shot_json(assistant_catalog) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/chat_terraria.py",
            "How do I craft Night's Edge?",
            "--database",
            str(assistant_catalog.database_path),
            "--json",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "found"
    assert payload["intent"] == "recipe"
    assert payload["facts"]["variant_count"] == 2


def test_pluggable_grounded_generator_receives_context(assistant_catalog) -> None:
    from src.assistant import CallableAnswerGenerator

    observed = {}

    def generate(context, fallback):
        observed["intent"] = context.intent.value
        observed["entity"] = context.entity
        observed["fallback"] = fallback
        return f"LLM-ready grounded answer for {context.entity}"

    with TerrariaAssistant(
        assistant_catalog.database_path,
        generator=CallableAnswerGenerator(generate),
    ) as assistant:
        response = assistant.answer("How do I craft Night's Edge?")

    assert response.answer == "LLM-ready grounded answer for Night's Edge"
    assert observed["intent"] == "recipe"
    assert observed["entity"] == "Night's Edge"
    assert "Volcano" in observed["fallback"]
