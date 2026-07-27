"""Explicit model baselines used for grounding ablations."""

from __future__ import annotations

from typing import Any, Mapping

from src.inference.chat_runtime import QwenPairRuntime

from .schemas import GameGuideResult


def build_ungrounded_messages(result: GameGuideResult) -> list[Mapping[str, str]]:
    """Build a no-evidence prompt for a controlled hallucination baseline.

    This prompt is intentionally not used by the deployed assistant. It exists
    only to measure what changes when the same Qwen checkpoint receives no
    retrieved facts, provenance, or deterministic decision.
    """

    is_chinese = any("\u4e00" <= char <= "\u9fff" for char in result.question)
    if is_chinese:
        system = (
            "你是一个游戏攻略助手。请直接回答用户问题，保持简洁、实用。"
            "这是无检索实验基线；不要声称你访问了数据库或来源。"
            "不要输出思维过程或 <think> 标签。"
        )
        user = f"游戏：{result.game}\n问题：{result.question}"
    else:
        system = (
            "You are a concise game-guide assistant. Answer the user's question "
            "directly. This is a no-retrieval experimental baseline; do not claim "
            "that you consulted a database or source. Do not emit chain-of-thought "
            "or <think> tags."
        )
        user = f"Game: {result.game}\nQuestion: {result.question}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


class UngroundedQwenGenerator:
    """Generate without evidence for an ablation baseline.

    No grounding validator or deterministic fallback is applied because doing
    so would turn this control condition back into a grounded system. Runtime
    failures still return the deterministic answer so evaluation jobs can
    finish and record the failure in debug metadata.
    """

    def __init__(
        self,
        runtime: QwenPairRuntime,
        *,
        engine: str | None = None,
        fallback_on_error: bool = True,
    ) -> None:
        self.runtime = runtime
        self.engine = engine
        self.fallback_on_error = bool(fallback_on_error)
        self.last_debug: dict[str, Any] = {}
        self.last_warnings: list[str] = []

    def generate(self, result: GameGuideResult) -> str:
        self.last_debug = {"baseline": "ungrounded"}
        self.last_warnings = []
        try:
            runtime_result = self.runtime.generate(
                build_ungrounded_messages(result),
                engine=self.engine,
            )
        except Exception as error:
            self.last_debug.update(
                {
                    "error": f"{type(error).__name__}: {error}",
                    "fallback_used": True,
                }
            )
            self.last_warnings.append(
                "Ungrounded baseline generation failed; deterministic answer used."
            )
            if self.fallback_on_error:
                return result.answer
            raise
        self.last_debug.update(
            {
                "runtime": runtime_result.to_dict(),
                "fallback_used": False,
            }
        )
        return runtime_result.text

    def close(self) -> None:
        self.runtime.close()
