
from __future__ import annotations

from src.evaluation.prompt_budget import analyze_prompt_budgets
from src.gameguide.schemas import GameEvidence, GameGuideResult


class PromptBudgetAssistant:
    def answer(self, question, *, game, player_state=None):
        evidence = [
            GameEvidence(
                source_id=f"S{index}",
                game=game,
                evidence_type="guide_chunk",
                source_catalog_id=f"guide:{index}",
                label=f"Guide {index}",
                source_url=f"https://example.test/{index}",
                page_title="Guide",
                section_title=f"Section {index}",
                score=1.0 - index * 0.05,
                payload={"text": "long evidence text " * 300},
            )
            for index in range(1, 9)
        ]
        return GameGuideResult(
            game=game,
            status="found",
            question=question,
            intent="guide",
            entity=None,
            answer="Fallback",
            facts={"hits": [
                {
                    "chunk_id": item.source_catalog_id,
                    "source_url": item.source_url,
                    "text": item.payload["text"],
                }
                for item in evidence
            ]},
            evidence=evidence,
        )


def test_prompt_budget_audit_reports_compaction() -> None:
    rows = analyze_prompt_budgets(
        PromptBudgetAssistant(),
        [{"id": "x", "game": "terraria", "question": "What next?"}],
        compact_sources=3,
        compact_characters=3500,
    )
    assert len(rows) == 1
    assert rows[0].compact_characters < rows[0].full_characters
    assert rows[0].compact_sources <= 3
    assert rows[0].character_reduction > 0.0
