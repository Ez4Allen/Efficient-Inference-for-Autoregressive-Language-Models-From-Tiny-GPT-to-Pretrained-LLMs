from __future__ import annotations

from dataclasses import dataclass

from src.gameguide import (
    GameEvidence,
    GameGuideAssistant,
    GameGuideQwenGenerator,
    GameGuideResult,
    build_gameguide_messages,
    validate_gameguide_answer,
)


class StubPlugin:
    game_id = "stub"
    display_name = "Stub"

    def answer(self, question, *, language="auto", player_state=None, include_debug=False):
        evidence = GameEvidence(
            source_id="S1",
            game="stub",
            evidence_type="fact",
            source_catalog_id="stub:1",
            label="Stub source",
            source_url="https://example.test/source",
        )
        return GameGuideResult(
            game="stub",
            status="found",
            question=question,
            intent="entity",
            entity="Thing",
            answer="Verified fallback [S1]",
            facts={"value": 1},
            evidence=[evidence],
        )

    def close(self):
        pass


@dataclass
class FakeRuntimeResult:
    text: str

    def to_dict(self):
        return {"text": self.text, "engine": "target"}


class FakeRuntime:
    def __init__(self, text):
        self.text = text

    def generate(self, messages, engine=None):
        return FakeRuntimeResult(self.text)

    def close(self):
        pass


def test_gameguide_prompt_is_game_agnostic():
    result = StubPlugin().answer("Question")
    messages = build_gameguide_messages(result)
    assert "GameGuideLM" in messages[0]["content"]
    assert "stub" in messages[1]["content"]
    assert "[S1]" in messages[1]["content"]


def test_validator_rejects_missing_citation():
    result = StubPlugin().answer("Question")
    validation = validate_gameguide_answer("An uncited factual answer.", result)
    assert validation.valid is False
    assert "missing_citations" in validation.issues


def test_qwen_generator_uses_shared_runtime_and_fallback():
    plugin = StubPlugin()
    generator = GameGuideQwenGenerator(FakeRuntime("Generated answer [S1]"))
    with GameGuideAssistant([plugin], generator=generator) as assistant:
        result = assistant.answer("Question", game="stub", include_debug=True)
    assert result.answer == "Generated answer [S1]"
    assert result.debug["generation"]["fallback_used"] is False


def test_qwen_generator_falls_back_on_invalid_output():
    generator = GameGuideQwenGenerator(FakeRuntime("No citation"))
    result = StubPlugin().answer("Question")
    assert generator.generate(result) == "Verified fallback [S1]"
    assert generator.last_debug["fallback_used"] is True


def test_validator_rejects_unsupported_number():
    result = StubPlugin().answer("Question")
    validation = validate_gameguide_answer("The value is 999. [S1]", result)
    assert validation.valid is False
    assert any(issue.startswith("unsupported_numbers:") for issue in validation.issues)


def test_validator_accepts_supported_number():
    result = StubPlugin().answer("Question")
    validation = validate_gameguide_answer("The value is 1. [S1]", result)
    assert validation.valid is True


def test_assistant_rejects_duplicate_plugins():
    plugin = StubPlugin()
    import pytest
    with pytest.raises(ValueError, match="Duplicate game plug-in ID"):
        GameGuideAssistant([plugin, plugin])


def test_assistant_rejects_empty_question():
    import pytest
    with GameGuideAssistant([StubPlugin()]) as assistant:
        with pytest.raises(ValueError, match="Question cannot be empty"):
            assistant.answer("   ", game="stub")


def test_assistant_rejects_calls_after_close():
    import pytest
    assistant = GameGuideAssistant([StubPlugin()])
    assistant.close()
    with pytest.raises(RuntimeError, match="closed"):
        assistant.answer("Question", game="stub")

def test_qwen_generator_respects_configured_max_answer_length():
    plugin = StubPlugin()
    generator = GameGuideQwenGenerator(
        FakeRuntime("Generated answer that is too long [S1]"),
        max_answer_chars=10,
    )
    result = plugin.answer("Question")
    assert generator.generate(result) == "Verified fallback [S1]"
    assert "answer_too_long" in generator.last_debug["validation"]["issues"]



def test_ungrounded_baseline_uses_no_evidence_prompt():
    from src.gameguide import UngroundedQwenGenerator, build_ungrounded_messages

    result = StubPlugin().answer("Question")
    messages = build_ungrounded_messages(result)
    assert "Evidence package" not in messages[1]["content"]
    assert "Question" in messages[1]["content"]

    generator = UngroundedQwenGenerator(FakeRuntime("Ungrounded answer"))
    assert generator.generate(result) == "Ungrounded answer"
    assert generator.last_debug["baseline"] == "ungrounded"
    assert generator.last_debug["fallback_used"] is False
