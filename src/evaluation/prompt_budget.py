
"""Audit model-input sizes before running expensive Qwen experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from src.gameguide.evidence_selection import EvidenceSelectionConfig
from src.gameguide.prompting import prepare_gameguide_prompt


@dataclass(slots=True)
class PromptBudgetRow:
    example_id: str
    game: str
    intent: str
    status: str
    full_characters: int
    compact_characters: int
    full_approximate_tokens: int
    compact_approximate_tokens: int
    character_reduction: float
    full_sources: int
    compact_sources: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _message_characters(messages) -> int:
    return sum(len(str(message.get("content", ""))) for message in messages)


def analyze_prompt_budgets(
    assistant: Any,
    annotations: Iterable[dict[str, Any]],
    *,
    compact_sources: int = 6,
    compact_characters: int = 14_000,
) -> list[PromptBudgetRow]:
    rows: list[PromptBudgetRow] = []
    for index, annotation in enumerate(annotations, start=1):
        game = annotation.get("game") or "terraria"
        if game == "stardew":
            game = "stardew_valley"
        result = assistant.answer(
            str(annotation["question"]),
            game=game,
            player_state=annotation.get("player_state"),
        )
        full = prepare_gameguide_prompt(
            result,
            evidence_config=EvidenceSelectionConfig(
                policy="full",
                max_sources=max(1, len(result.evidence) or 1),
                max_characters=10_000_000,
                max_characters_per_guide=10_000_000,
            ),
        )
        compact = prepare_gameguide_prompt(
            result,
            evidence_config=EvidenceSelectionConfig(
                policy="compact",
                max_sources=compact_sources,
                max_characters=compact_characters,
            ),
        )
        full_chars = _message_characters(full.messages)
        compact_chars = _message_characters(compact.messages)
        reduction = 1.0 - compact_chars / full_chars if full_chars else 0.0
        rows.append(
            PromptBudgetRow(
                example_id=str(annotation.get("id") or f"example_{index:06d}"),
                game=game,
                intent=result.intent,
                status=result.status,
                full_characters=full_chars,
                compact_characters=compact_chars,
                full_approximate_tokens=max(1, (full_chars + 3) // 4),
                compact_approximate_tokens=max(1, (compact_chars + 3) // 4),
                character_reduction=round(reduction, 6),
                full_sources=len(full.result.evidence),
                compact_sources=len(compact.result.evidence),
            )
        )
    return rows
