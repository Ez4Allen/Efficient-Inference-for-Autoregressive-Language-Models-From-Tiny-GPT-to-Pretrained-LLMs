
"""Build evidence-conditioned multi-game SFT examples."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from src.gameguide.evidence_selection import EvidenceSelectionConfig
from src.gameguide.prompting import PromptMode, prepare_gameguide_prompt
from src.gameguide.schemas import GameGuideResult
from src.utils.io import read_jsonl, write_jsonl


def ensure_citation(answer: str, result: GameGuideResult) -> str:
    """Attach a valid source label only when a found example has evidence."""

    text = str(answer).strip()
    if result.status == "found" and result.evidence and "[S" not in text:
        text += "\n\n[S1]"
    return text


def build_grounded_sft_record(
    *,
    example_id: str,
    result: GameGuideResult,
    target_answer: str,
    split: str,
    category: str | None = None,
    required_facts: list[Any] | None = None,
    forbidden_errors: list[str] | None = None,
    prompt_mode: PromptMode = "evidence_only",
    evidence_config: EvidenceSelectionConfig | None = None,
    target_source: str = "reviewed_reference",
) -> dict[str, Any]:
    """Create a reproducible evidence-conditioned chat record.

    The default input excludes the deterministic renderer answer. This makes
    the model learn evidence following instead of paraphrasing a prewritten
    response. ``scaffolded`` remains an explicit ablation condition.
    """

    prepared = prepare_gameguide_prompt(
        result,
        prompt_mode=prompt_mode,
        evidence_config=evidence_config,
    )
    messages = [dict(message) for message in prepared.messages]
    messages.append(
        {
            "role": "assistant",
            "content": ensure_citation(target_answer, prepared.result),
        }
    )
    return {
        "id": example_id,
        "game": result.game,
        "split": split,
        "category": category,
        "messages": messages,
        "intent": result.intent,
        "entity": result.entity,
        "retrieval_status": result.status,
        "prompt_mode": prompt_mode,
        "evidence_policy": prepared.evidence_report.policy,
        "evidence_selection": prepared.evidence_report.to_dict(),
        "evidence_ids": [item.source_id for item in prepared.result.evidence],
        "source_urls": [
            item.source_url for item in prepared.result.evidence if item.source_url
        ],
        "required_facts": required_facts or [],
        "forbidden_errors": forbidden_errors or [],
        "target_source": target_source,
        "dataset_version": "gameguidelm-grounded-v2",
    }


def build_refusal_sft_record(
    *,
    example_id: str,
    result: GameGuideResult,
    split: str,
    category: str | None = None,
) -> dict[str, Any]:
    """Preserve deterministic refusal/clarification decisions as supervision."""

    return build_grounded_sft_record(
        example_id=example_id,
        result=result,
        target_answer=result.answer,
        split=split,
        category=category,
        prompt_mode="evidence_only",
        evidence_config=EvidenceSelectionConfig(policy="compact"),
        target_source="deterministic_decision",
    )


def extract_annotation_request(
    record: dict[str, Any],
    *,
    default_game: str | None = None,
    record_label: str = "record",
) -> tuple[str, str]:
    """Extract a game and user question from annotation or chat-SFT records."""

    game_value = record.get("game") or record.get("domain") or default_game
    game = str(game_value or "").strip()
    if game == "stardew":
        game = "stardew_valley"
    if not game:
        raise ValueError(f"{record_label} has no game/domain value.")

    direct_question = record.get("question")
    if isinstance(direct_question, str) and direct_question.strip():
        return game, direct_question.strip()

    messages = record.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return game, content.strip()

    raise ValueError(
        f"{record_label} has neither a non-empty question nor a user message."
    )


def load_annotation_records(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(read_jsonl(Path(path)))
    return records


def write_grounded_sft(records: list[dict[str, Any]], path: str | Path) -> None:
    write_jsonl(Path(path), records)
