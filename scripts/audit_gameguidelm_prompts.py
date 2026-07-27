
#!/usr/bin/env python3
"""Compare full and compact evidence prompt budgets without loading Qwen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.gameguide_eval import normalize_annotation
from src.evaluation.prompt_budget import analyze_prompt_budgets
from src.gameguide import GameGuideAssistant
from src.games.stardew import StardewAssistant
from src.games.terraria import TerrariaGamePlugin
from src.utils.io import read_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--default-game", choices=("terraria", "stardew_valley"))
    parser.add_argument("--max-evidence-sources", type=int, default=6)
    parser.add_argument("--max-evidence-characters", type=int, default=14_000)
    args = parser.parse_args()

    annotations = []
    for path in args.input:
        for record in read_jsonl(path):
            annotations.append(
                normalize_annotation(
                    record,
                    source_path=path,
                    default_game=args.default_game,
                )
            )

    with GameGuideAssistant(
        [TerrariaGamePlugin(auto_build=True), StardewAssistant(auto_build=True)]
    ) as assistant:
        rows = analyze_prompt_budgets(
            assistant,
            annotations,
            compact_sources=args.max_evidence_sources,
            compact_characters=args.max_evidence_characters,
        )

    payloads = [row.to_dict() for row in rows]
    write_jsonl(args.output, payloads)
    total = len(rows)
    summary = {
        "examples": total,
        "mean_full_characters": sum(row.full_characters for row in rows) / total if total else 0.0,
        "mean_compact_characters": sum(row.compact_characters for row in rows) / total if total else 0.0,
        "mean_character_reduction": sum(row.character_reduction for row in rows) / total if total else 0.0,
        "max_full_characters": max((row.full_characters for row in rows), default=0),
        "max_compact_characters": max((row.compact_characters for row in rows), default=0),
    }
    write_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
