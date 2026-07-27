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
