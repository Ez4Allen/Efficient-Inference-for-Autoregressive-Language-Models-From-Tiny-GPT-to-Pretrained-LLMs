from __future__ import annotations

from src.gameguide.schemas import GameEvidence, GameGuideResult
from src.training.grounded_sft import build_grounded_sft_record


def test_grounded_sft_record_contains_evidence_prompt_and_target():
    result = GameGuideResult(
        game="stardew_valley",
        status="found",
        question="How long does Parsnip take?",
        intent="crop_info",
        entity="Parsnip",
        answer="Four days [S1]",
        facts={"growth_days": 4},
        evidence=[
            GameEvidence(
                source_id="S1",
                game="stardew_valley",
                evidence_type="crop",
                source_catalog_id="stardew:crop:parsnip",
                label="Parsnip",
            )
        ],
    )
    record = build_grounded_sft_record(
        example_id="x",
        result=result,
        target_answer="Parsnip takes four days.",
        split="train",
    )
    assert [message["role"] for message in record["messages"]] == ["system", "user", "assistant"]
    assert "[S1]" in record["messages"][-1]["content"]
    assert record["game"] == "stardew_valley"


def test_extract_annotation_request_supports_chat_sft_records() -> None:
    from src.training.grounded_sft import extract_annotation_request

    game, question = extract_annotation_request(
        {
            "domain": "stardew_valley",
            "messages": [
                {"role": "system", "content": "Be grounded."},
                {"role": "user", "content": "Where can I catch Catfish?"},
                {"role": "assistant", "content": "In rivers."},
            ],
        }
    )

    assert game == "stardew_valley"
    assert question == "Where can I catch Catfish?"


def test_extract_annotation_request_prefers_direct_question_and_default_game() -> None:
    from src.training.grounded_sft import extract_annotation_request

    game, question = extract_annotation_request(
        {"question": "  What should I do next?  "},
        default_game="terraria",
    )

    assert game == "terraria"
    assert question == "What should I do next?"
