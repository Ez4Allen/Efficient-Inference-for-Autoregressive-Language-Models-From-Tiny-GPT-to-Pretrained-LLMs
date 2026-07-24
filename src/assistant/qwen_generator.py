"""Qwen-backed grounded answer generator using the shared inference runtime."""

from __future__ import annotations

from typing import Any

from src.inference.chat_runtime import QwenPairRuntime

from .answer_validator import validate_grounded_answer
from .prompt_templates import build_grounded_messages
from .schemas import ContextBundle


class QwenGroundedAnswerGenerator:
    """Generate a cited answer and fall back on any unsafe or invalid output."""

    def __init__(
        self,
        runtime: QwenPairRuntime,
        *,
        engine: str | None = None,
        require_citations: bool = True,
        fallback_on_error: bool = True,
        max_answer_chars: int = 6000,
    ) -> None:
        self.runtime = runtime
        self.engine = engine
        self.require_citations = bool(require_citations)
        self.fallback_on_error = bool(fallback_on_error)
        self.max_answer_chars = int(max_answer_chars)
        self.last_debug: dict[str, Any] = {}
        self.last_warnings: list[str] = []

    def generate(self, context: ContextBundle, fallback_answer: str) -> str:
        self.last_debug = {}
        self.last_warnings = []

        if context.payload.get("status") != "found":
            self.last_debug = {
                "skipped": True,
                "reason": "retrieval_status_is_not_found",
                "fallback_used": True,
            }
            return fallback_answer

        messages = build_grounded_messages(context, fallback_answer)
        try:
            result = self.runtime.generate(messages, engine=self.engine)
        except Exception as error:
            self.last_debug = {
                "skipped": False,
                "error": f"{type(error).__name__}: {error}",
                "fallback_used": True,
            }
            self.last_warnings.append(
                "LLM generation failed; the deterministic grounded answer was used."
            )
            if self.fallback_on_error:
                return fallback_answer
            raise

        validation = validate_grounded_answer(
            result.text,
            context,
            require_citations=self.require_citations,
            max_answer_chars=self.max_answer_chars,
        )
        self.last_debug = {
            "skipped": False,
            "runtime": result.to_dict(),
            "validation": validation.to_dict(),
            "fallback_used": not validation.valid,
        }
        if not validation.valid:
            self.last_warnings.append(
                "LLM output failed grounding validation; the deterministic answer was used."
            )
            return fallback_answer
        return result.text

    def close(self) -> None:
        self.runtime.close()
