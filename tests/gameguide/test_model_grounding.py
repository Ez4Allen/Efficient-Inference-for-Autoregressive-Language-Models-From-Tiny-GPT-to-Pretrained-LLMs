
from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.evaluation.experiment_matrix import load_experiment_matrix
from src.gameguide import (
    EvidenceSelectionConfig,
    GameEvidence,
    GameGuideQwenGenerator,
    GameGuideResult,
    build_gameguide_messages,
    prepare_evidence,
)
from src.models.runtime_config import load_qwen_pair_config


def make_result(*, long_guides: bool = False) -> GameGuideResult:
    evidence = [
        GameEvidence(
            source_id="S1",
            game="terraria",
            evidence_type="recipe",
            source_catalog_id="recipe:nights-edge",
            label="Night's Edge recipe",
            payload={"ingredients": ["Muramasa", "Volcano"]},
        ),
        GameEvidence(
            source_id="S2",
            game="terraria",
            evidence_type="guide_chunk",
            source_catalog_id="guide:progression:1",
            label="Progression guide",
            source_url="https://example.test/guide",
            page_title="Guide:Progression",
            section_title="Early game",
            score=0.9,
            payload={"text": "guide evidence " * (500 if long_guides else 4)},
        ),
        GameEvidence(
            source_id="S3",
            game="terraria",
            evidence_type="guide_chunk",
            source_catalog_id="guide:progression:2",
            label="Second guide",
            source_url="https://example.test/guide2",
            page_title="Guide:Progression",
            section_title="Later game",
            score=0.5,
            payload={"text": "secondary evidence " * (500 if long_guides else 4)},
        ),
    ]
    return GameGuideResult(
        game="terraria",
        status="found",
        question="How do I craft Night's Edge?",
        intent="recipe",
        entity="Night's Edge",
        answer="DETERMINISTIC FALLBACK ANSWER [S1]",
        facts={
            "ingredients": ["Muramasa", "Volcano"],
            "hits": [
                {
                    "chunk_id": "guide:progression:1",
                    "source_url": "https://example.test/guide",
                    "text": "guide evidence " * (500 if long_guides else 4),
                },
                {
                    "chunk_id": "guide:progression:2",
                    "source_url": "https://example.test/guide2",
                    "text": "secondary evidence " * (500 if long_guides else 4),
                },
            ],
        },
        evidence=evidence,
    )


def test_default_prompt_is_evidence_only() -> None:
    messages = build_gameguide_messages(make_result())
    assert "DETERMINISTIC FALLBACK ANSWER" not in messages[1]["content"]
    assert "Muramasa" in messages[1]["content"]
    assert "[S1]" in messages[1]["content"]


def test_scaffolded_prompt_is_an_explicit_ablation() -> None:
    messages = build_gameguide_messages(make_result(), prompt_mode="scaffolded")
    assert "DETERMINISTIC FALLBACK ANSWER" in messages[1]["content"]
    assert "scaffold" in messages[1]["content"].casefold()


def test_compact_evidence_reduces_source_count_and_prompt_size() -> None:
    result = make_result(long_guides=True)
    prepared = prepare_evidence(
        result,
        EvidenceSelectionConfig(
            policy="compact",
            max_sources=2,
            max_characters=3000,
            max_characters_per_guide=800,
        ),
    )
    assert prepared.report.selected_sources <= 2
    assert prepared.report.selected_characters < prepared.report.original_characters
    assert prepared.report.selected_source_ids[0] == "S1"


def test_structured_only_evidence_excludes_guide_chunks() -> None:
    prepared = prepare_evidence(
        make_result(),
        EvidenceSelectionConfig(policy="structured_only", max_sources=6),
    )
    assert [item.source_id for item in prepared.result.evidence] == ["S1"]


@dataclass
class FakeRuntimeResult:
    text: str

    def to_dict(self):
        return {"text": self.text, "engine": "target", "prompt_tokens": 10}


class SequenceRuntime:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def generate(self, messages, engine=None):
        index = min(self.calls, len(self.outputs) - 1)
        self.calls += 1
        return FakeRuntimeResult(self.outputs[index])

    def close(self):
        pass


def test_generator_repairs_an_uncited_first_answer() -> None:
    runtime = SequenceRuntime(["Uncited answer", "Supported answer [S1]"])
    generator = GameGuideQwenGenerator(runtime, max_repair_attempts=1)
    answer = generator.generate(make_result())
    assert answer == "Supported answer [S1]"
    assert runtime.calls == 2
    assert generator.last_debug["repair_used"] is True
    assert generator.last_debug["fallback_used"] is False


def test_generator_falls_back_after_failed_repair() -> None:
    runtime = SequenceRuntime(["Uncited answer", "Still uncited"])
    generator = GameGuideQwenGenerator(runtime, max_repair_attempts=1)
    answer = generator.generate(make_result())
    assert answer.startswith("DETERMINISTIC FALLBACK")
    assert runtime.calls == 2
    assert generator.last_debug["fallback_used"] is True


def test_runtime_config_exposes_model_input_controls(tmp_path) -> None:
    path = tmp_path / "pair.yaml"
    path.write_text(
        """
models:
  draft: {model_name_or_path: Qwen/Qwen3-0.6B}
  target: {model_name_or_path: Qwen/Qwen3-4B}
grounding:
  prompt_mode: scaffolded
  evidence_policy: structured_only
  max_evidence_sources: 3
  max_evidence_characters: 5000
  max_repair_attempts: 2
""",
        encoding="utf-8",
    )
    config = load_qwen_pair_config(path)
    assert config.grounding.prompt_mode == "scaffolded"
    assert config.grounding.evidence_policy == "structured_only"
    assert config.grounding.max_evidence_sources == 3
    assert config.grounding.max_evidence_characters == 5000
    assert config.grounding.max_repair_attempts == 2


def test_runtime_config_rejects_invalid_evidence_policy(tmp_path) -> None:
    path = tmp_path / "pair.yaml"
    path.write_text(
        """
models:
  draft: {model_name_or_path: draft}
  target: {model_name_or_path: target}
grounding:
  evidence_policy: invented_policy
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="evidence_policy"):
        load_qwen_pair_config(path)


def test_experiment_matrix_contains_required_ablation_conditions() -> None:
    matrix = load_experiment_matrix("configs/gameguidelm_experiments.yaml")
    names = {condition.name for condition in matrix.conditions}
    assert {
        "deterministic_grounded",
        "ungrounded_target",
        "grounded_full_target",
        "grounded_compact_target",
        "scaffolded_compact_target",
        "grounded_compact_speculative",
    }.issubset(names)
