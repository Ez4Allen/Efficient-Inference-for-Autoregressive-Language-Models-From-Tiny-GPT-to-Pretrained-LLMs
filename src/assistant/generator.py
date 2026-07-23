"""Pluggable grounded-answer generation interfaces."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from .schemas import ContextBundle


@runtime_checkable
class GroundedAnswerGenerator(Protocol):
    """Generate an answer from grounded context with a deterministic fallback."""

    def generate(self, context: ContextBundle, fallback_answer: str) -> str:
        ...


class CallableAnswerGenerator:
    """Adapt a plain callable to :class:`GroundedAnswerGenerator`.

    The callable receives the complete :class:`ContextBundle` and the
    deterministic renderer output.  It can invoke a local model, remote API, or
    another template system.  Empty output is rejected so callers always have a
    usable grounded answer.
    """

    def __init__(
        self,
        function: Callable[[ContextBundle, str], str],
    ) -> None:
        if not callable(function):
            raise TypeError("function must be callable.")
        self.function = function

    def generate(self, context: ContextBundle, fallback_answer: str) -> str:
        answer = str(self.function(context, fallback_answer)).strip()
        if not answer:
            raise ValueError("Grounded answer generator returned empty output.")
        return answer
