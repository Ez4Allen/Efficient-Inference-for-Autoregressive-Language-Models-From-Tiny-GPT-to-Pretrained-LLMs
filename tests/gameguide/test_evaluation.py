from __future__ import annotations

from src.evaluation.gameguide_eval import infer_game_from_path, normalize_annotation, score_gameguide_result
from src.gameguide.schemas import GameEvidence, GameGuideResult


def test_evaluation_scores_required_and_forbidden_facts():
    result = GameGuideResult(
        game="stardew_valley",
        status="found",
        question="How long?",
        intent="crop_info",
        entity="Parsnip",
        answer="Parsnip takes 4 days in Spring. [S1]",
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
    annotation = {
        "id": "x",
        "game": "stardew_valley",
        "intent": "crop_info",
        "expected_status": "found",
        "must_include": ["4 days", "Spring"],
        "must_not_include": ["5 days"],
    }
    score = score_gameguide_result(annotation, result)
    assert score.passed is True
    assert score.required_fact_coverage == 1.0
    assert score.forbidden_error_rate == 0.0

def test_legacy_terraria_annotation_infers_game_from_path():
    record = normalize_annotation(
        {"id": "legacy", "question": "What happens after Wall of Flesh?"},
        source_path="data/terraria/terraria_eval.jsonl",
    )
    assert record["game"] == "terraria"


def test_game_inference_does_not_guess_from_question_text():
    assert infer_game_from_path("data/custom/eval.jsonl") is None

def test_evaluation_normalizes_spacing_apostrophes_and_clock_format():
    result = GameGuideResult(
        game="stardew_valley",
        status="found",
        question="When?",
        intent="fish_availability",
        entity="Midnight Carp",
        answer="Available in Fall from 10:00 PM to 2:00 AM at Leah's Cottage. 8 天。",
    )
    annotation = {
        "id": "normalize",
        "game": "stardew_valley",
        "intent": "fish_availability",
        "expected_status": "found",
        "must_include": ["Fall", "10pm", "2am", "Leah’s Cottage", "8天"],
    }
    score = score_gameguide_result(annotation, result)
    assert score.required_fact_coverage == 1.0


def test_non_found_status_does_not_depend_on_specific_refusal_wording():
    result = GameGuideResult(
        game="stardew_valley",
        status="not_found",
        question="Where?",
        intent="fish_availability",
        entity="Galaxy Catfish",
        answer="No supported local evidence was found.",
    )
    annotation = {
        "id": "not-found",
        "game": "stardew_valley",
        "intent": "fish_availability",
        "expected_status": "not_found",
        "must_include": ["not found"],
    }
    assert score_gameguide_result(annotation, result).passed is True


def test_evaluation_summary_includes_model_and_slice_metrics():
    from src.evaluation.gameguide_eval import evaluate_annotations

    class StubAssistant:
        def answer(self, question, *, game, player_state=None, include_debug=False):
            return GameGuideResult(
                game=game,
                status="found",
                question=question,
                intent="crop_info",
                entity="Parsnip",
                answer="Parsnip takes 4 days. [S1]",
                evidence=[
                    GameEvidence(
                        source_id="S1",
                        game=game,
                        evidence_type="crop",
                        source_catalog_id="stardew:crop:parsnip",
                        label="Parsnip",
                    )
                ],
                debug={
                    "generation": {
                        "fallback_used": False,
                        "evidence_selection": {"selected_sources": 1},
                        "runtime": {
                            "prompt_tokens": 128,
                            "generated_tokens": 12,
                            "ttft_seconds": 0.2,
                            "total_time_seconds": 0.8,
                        },
                        "validation": {"valid": True, "issues": []},
                    }
                },
            )

    annotations = [
        {
            "id": "eval-1",
            "game": "stardew_valley",
            "language": "en",
            "question": "How long does Parsnip take?",
            "intent": "crop_info",
            "expected_status": "found",
            "must_include": ["4 days"],
        }
    ]
    rows, summary = evaluate_annotations(StubAssistant(), annotations)
    assert rows[0]["score"]["citation_valid"] is True
    assert summary["mean_prompt_tokens"] == 128.0
    assert summary["fallback_rate"] == 0.0
    assert summary["slices"]["game"]["stardew_valley"]["examples"] == 1
    assert summary["slices"]["language"]["en"]["pass_rate"] == 1.0
