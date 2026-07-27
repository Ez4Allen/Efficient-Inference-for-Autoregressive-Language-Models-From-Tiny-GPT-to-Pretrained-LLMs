"""Qwen-backed answer generation for any registered game plug-in."""

from __future__ import annotations

from typing import Any

from src.inference.chat_runtime import QwenPairRuntime

from .evidence_selection import EvidenceSelectionConfig
from .prompting import PromptMode, build_repair_messages, prepare_gameguide_prompt
from .schemas import GameGuideResult
from .validation import validate_gameguide_answer


class GameGuideQwenGenerator:
    """Generate a cited answer and fall back on unsafe output.

    The Qwen runtime is shared with the autoregressive and speculative-decoding
    experiments. The generator owns prompt construction, a bounded repair
    attempt, and grounding validation; it never loads a second model stack.
    """

    def __init__(
        self,
        runtime: QwenPairRuntime,
        *,
        engine: str | None = None,
        require_citations: bool = True,
        validate_numbers: bool = True,
        fallback_on_error: bool = True,
        max_answer_chars: int = 7000,
        prompt_mode: PromptMode = "evidence_only",
        evidence_config: EvidenceSelectionConfig | None = None,
        max_repair_attempts: int = 1,
    ) -> None:
        self.runtime = runtime
        self.engine = engine
        self.require_citations = bool(require_citations)
        self.validate_numbers = bool(validate_numbers)
        self.fallback_on_error = bool(fallback_on_error)
        self.max_answer_chars = int(max_answer_chars)
        self.prompt_mode = prompt_mode
        self.evidence_config = evidence_config or EvidenceSelectionConfig()
        self.max_repair_attempts = int(max_repair_attempts)
        if self.max_answer_chars <= 0:
            raise ValueError("max_answer_chars must be positive.")
        if self.prompt_mode not in {"evidence_only", "scaffolded"}:
            raise ValueError("prompt_mode must be evidence_only or scaffolded.")
        if self.max_repair_attempts < 0 or self.max_repair_attempts > 2:
            raise ValueError("max_repair_attempts must be between 0 and 2.")
        self.last_debug: dict[str, Any] = {}
        self.last_warnings: list[str] = []

    def _validate(self, text: str, prepared_result: GameGuideResult):
        return validate_gameguide_answer(
            text,
            prepared_result,
            require_citations=self.require_citations,
            validate_numbers=self.validate_numbers,
            max_answer_chars=self.max_answer_chars,
        )

    def generate(self, result: GameGuideResult) -> str:
        self.last_debug = {}
        self.last_warnings = []
        if result.status != "found":
            self.last_debug = {
                "skipped": True,
                "reason": f"status:{result.status}",
                "fallback_used": True,
            }
            return result.answer

        prepared = prepare_gameguide_prompt(
            result,
            prompt_mode=self.prompt_mode,
            evidence_config=self.evidence_config,
        )
        attempts: list[dict[str, Any]] = []
        messages = prepared.messages

        for attempt_index in range(self.max_repair_attempts + 1):
            try:
                runtime_result = self.runtime.generate(messages, engine=self.engine)
            except Exception as error:
                self.last_debug = {
                    "skipped": False,
                    "prompt_mode": self.prompt_mode,
                    "evidence_selection": prepared.evidence_report.to_dict(),
                    "attempts": attempts,
                    "error": f"{type(error).__name__}: {error}",
                    "fallback_used": True,
                }
                self.last_warnings.append(
                    "LLM generation failed; deterministic evidence answer used."
                )
                if self.fallback_on_error:
                    return result.answer
                raise

            validation = self._validate(runtime_result.text, prepared.result)
            attempts.append({
                "attempt": attempt_index + 1,
                "runtime": runtime_result.to_dict(),
                "validation": validation.to_dict(),
            })
            if validation.valid:
                self.last_debug = {
                    "skipped": False,
                    "prompt_mode": self.prompt_mode,
                    "evidence_selection": prepared.evidence_report.to_dict(),
                    "attempts": attempts,
                    "runtime": runtime_result.to_dict(),
                    "validation": validation.to_dict(),
                    "repair_used": attempt_index > 0,
                    "fallback_used": False,
                }
                if attempt_index > 0:
                    self.last_warnings.append(
                        "The first model answer failed grounding validation and was repaired."
                    )
                return runtime_result.text

            if attempt_index < self.max_repair_attempts:
                messages = build_repair_messages(
                    prepared,
                    runtime_result.text,
                    validation.issues,
                )

        self.last_debug = {
            "skipped": False,
            "prompt_mode": self.prompt_mode,
            "evidence_selection": prepared.evidence_report.to_dict(),
            "attempts": attempts,
            "runtime": attempts[-1]["runtime"] if attempts else None,
            "validation": attempts[-1]["validation"] if attempts else None,
            "repair_used": len(attempts) > 1,
            "fallback_used": True,
        }
        self.last_warnings.append(
            "LLM output failed grounding validation; deterministic evidence answer used."
        )
        return result.answer

    def close(self) -> None:
        self.runtime.close()
