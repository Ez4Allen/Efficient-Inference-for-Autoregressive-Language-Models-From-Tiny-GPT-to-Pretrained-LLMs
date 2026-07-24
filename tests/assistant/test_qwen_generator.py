from __future__ import annotations

from dataclasses import dataclass

from src.assistant.answer_validator import validate_grounded_answer
from src.assistant.qwen_generator import QwenGroundedAnswerGenerator
from src.assistant.schemas import AssistantIntent, ContextBundle
from src.inference.chat_runtime import ChatGenerationResult


@dataclass
class FakeRuntime:
    text: str
    calls: int = 0

    def generate(self, messages, *, engine=None):
        self.calls += 1
        assert messages[0]["role"] == "system"
        return ChatGenerationResult(
            text=self.text,
            engine=engine or "target",
            prompt_tokens=20,
            generated_tokens=8,
            total_time_seconds=0.2,
            ttft_seconds=0.05,
            mean_tpot_seconds=0.02,
            tokens_per_second=40.0,
            target_forward_calls=8,
        )

    def close(self):
        return None


def context(status: str = "found") -> ContextBundle:
    evidence = [
        {
            "source_id": "S1",
            "source_catalog_id": "item:1",
            "source_url": "https://example.test/item/1",
        }
    ]
    return ContextBundle(
        intent=AssistantIntent.ITEM,
        entity="Night's Edge",
        language="en",
        text="grounded context",
        payload={
            "question": "What is Night's Edge?",
            "status": status,
            "facts": {"name": "Night's Edge"},
            "evidence": evidence,
        },
        evidence=evidence,
    )


def test_validator_accepts_only_known_citations() -> None:
    valid = validate_grounded_answer("It is a sword. [S1]", context())
    assert valid.valid

    invalid = validate_grounded_answer("It is a sword. [S9]", context())
    assert not invalid.valid
    assert any(issue.startswith("invalid_citations") for issue in invalid.issues)


def test_generator_returns_valid_model_answer() -> None:
    runtime = FakeRuntime("Night's Edge is a sword. [S1]")
    generator = QwenGroundedAnswerGenerator(runtime, engine="target")

    answer = generator.generate(context(), "verified fallback")

    assert answer == "Night's Edge is a sword. [S1]"
    assert runtime.calls == 1
    assert generator.last_debug["fallback_used"] is False


def test_generator_falls_back_for_uncited_or_missing_evidence() -> None:
    runtime = FakeRuntime("Night's Edge is a sword.")
    generator = QwenGroundedAnswerGenerator(runtime)

    assert generator.generate(context(), "verified fallback") == "verified fallback"
    assert generator.last_debug["fallback_used"] is True

    assert generator.generate(context("not_found"), "not found fallback") == "not found fallback"
    assert runtime.calls == 1
