from __future__ import annotations

from src.evaluation.gameguide_eval import build_answer_validation_trace
from src.gameguide.schemas import GameEvidence, GameGuideResult


def test_validation_trace_explains_pass_formula() -> None:
    annotation = {
        "id": "case",
        "game": "stardew_valley",
        "language": "en",
        "intent": "fish_availability",
        "expected_status": "found",
        "required_facts": ["Spring", "rain", "river"],
        "forbidden_errors": ["Summer only"],
    }
    result = GameGuideResult(
        game="stardew_valley",
        status="found",
        question="When can I catch Catfish?",
        intent="fish_availability",
        entity="Catfish",
        answer="Catfish is found in the river in Spring during rain. [S1]",
        evidence=[
            GameEvidence(
                source_id="S1",
                game="stardew_valley",
                evidence_type="fish",
                source_catalog_id="catfish",
                label="Catfish",
            )
        ],
        debug={
            "generation": {
                "runtime": {"prompt_tokens": 10, "generated_tokens": 12},
                "validation": {"issues": []},
                "fallback_used": False,
            }
        },
    )
    trace = build_answer_validation_trace(annotation, result)
    assert trace.passed is True
    assert trace.required_fact_coverage == 1.0
    assert all(item.matched for item in trace.required_fact_traces)
    assert "coverage" in trace.pass_formula
