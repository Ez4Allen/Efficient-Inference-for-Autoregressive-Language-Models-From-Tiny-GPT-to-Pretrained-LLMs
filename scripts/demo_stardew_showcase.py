#!/usr/bin/env python3
"""Run a deterministic bilingual Stardew Valley showcase and export results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.games.stardew import StardewAssistant
from src.utils.io import write_json

DEFAULT_CASES: list[dict[str, Any]] = [
    {"id": "crop_en", "label": "Structured crop fact", "question": "How long does Parsnip take to grow?"},
    {"id": "crop_zh", "label": "中文作物查询", "question": "Strawberry收获后会再生吗？"},
    {"id": "deadline", "label": "Player-state calculation", "question": "秋季第15天种Pumpkin还能收获吗？"},
    {"id": "fish", "label": "Conditional fish availability", "question": "Legend在哪里、什么时候能钓？"},
    {"id": "gift", "label": "Villager gift lookup", "question": "阿比盖尔喜欢什么礼物？"},
    {"id": "recipe", "label": "Recipe lookup", "question": "Keg怎么制作？"},
    {"id": "bundle", "label": "Standard bundle", "question": "What is required for the River Fish Bundle?"},
    {
        "id": "bundle_partial",
        "label": "Version/mode-safe partial answer",
        "question": "What is in the remixed River Fish Bundle?",
        "player_state": {"bundle_mode": "remixed"},
    },
    {"id": "acquisition", "label": "Acquisition relation", "question": "Where can I buy Return Scepter?"},
    {"id": "guide", "label": "Guide retrieval", "question": "沙漠矿洞应该怎么准备？"},
    {"id": "context", "label": "Missing-context refusal", "question": "What should I plant today?"},
    {"id": "false_premise", "label": "False-premise refusal", "question": "Where can I get Dragon Tractor?"},
]


def run_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    with StardewAssistant(auto_build=True) as assistant:
        for case in cases:
            result = assistant.answer(
                case["question"],
                player_state=case.get("player_state"),
                include_debug=True,
            )
            outputs.append({
                "id": case["id"],
                "label": case["label"],
                "question": case["question"],
                "player_state": case.get("player_state") or {},
                "status": result.status,
                "intent": result.intent,
                "entity": result.entity,
                "answer": result.answer,
                "evidence": [
                    {
                        "source_id": item.source_id,
                        "label": item.label,
                        "source_catalog_id": item.source_catalog_id,
                        "source_url": item.source_url,
                        "page_title": item.page_title,
                        "section_title": item.section_title,
                        "score": item.score,
                    }
                    for item in result.evidence
                ],
                "route": (result.debug or {}).get("route"),
            })
    return outputs


def markdown(outputs: list[dict[str, Any]]) -> str:
    lines = ["# Stardew Valley deterministic showcase", ""]
    for index, row in enumerate(outputs, start=1):
        lines.extend([
            f"## {index}. {row['label']}",
            "",
            f"**Question:** {row['question']}",
            "",
            f"**Status / intent:** `{row['status']}` / `{row['intent']}`",
            "",
            f"**Answer:** {row['answer']}",
            "",
        ])
        if row["evidence"]:
            lines.append("**Evidence:**")
            for source in row["evidence"]:
                lines.append(f"- {source['source_id']}: {source['label']} — {source.get('source_url') or source['source_catalog_id']}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "stardew" / "demo_outputs.json")
    parser.add_argument("--markdown", type=Path, default=PROJECT_ROOT / "results" / "stardew" / "demo_outputs.md")
    args = parser.parse_args()
    outputs = run_cases(DEFAULT_CASES)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, {"game": "stardew_valley", "examples": outputs})
    args.markdown.write_text(markdown(outputs), encoding="utf-8")
    print(json.dumps({"status": "passed", "examples": len(outputs), "output": str(args.output), "markdown": str(args.markdown)}, indent=2))


if __name__ == "__main__":
    main()
