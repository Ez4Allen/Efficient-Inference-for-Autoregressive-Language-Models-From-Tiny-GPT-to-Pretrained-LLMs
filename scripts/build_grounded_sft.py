
#!/usr/bin/env python3
"""Convert reviewed QA annotations into evidence-conditioned SFT JSONL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gameguide import EvidenceSelectionConfig, GameGuideAssistant
from src.games.stardew import StardewAssistant
from src.games.terraria import TerrariaGamePlugin
from src.training.grounded_sft import (
    build_grounded_sft_record,
    load_annotation_records,
    write_grounded_sft,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--default-game", choices=("terraria", "stardew_valley"))
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--prompt-mode",
        choices=("evidence_only", "scaffolded"),
        default="evidence_only",
    )
    parser.add_argument(
        "--evidence-policy",
        choices=("compact", "full", "structured_only", "guide_only"),
        default="compact",
    )
    parser.add_argument("--max-evidence-sources", type=int, default=6)
    parser.add_argument("--max-evidence-characters", type=int, default=14_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_annotation_records(args.input)
    evidence_config = EvidenceSelectionConfig(
        policy=args.evidence_policy,
        max_sources=args.max_evidence_sources,
        max_characters=args.max_evidence_characters,
    )
    output = []
    with GameGuideAssistant(
        [TerrariaGamePlugin(auto_build=True), StardewAssistant(auto_build=True)]
    ) as assistant:
        for index, record in enumerate(records, start=1):
            game = record.get("game") or args.default_game
            if not game:
                raise ValueError(
                    f"Record {index} has no game and --default-game was not supplied."
                )
            game = "stardew_valley" if game == "stardew" else game
            question = str(record.get("question", "")).strip()
            answer = str(record.get("reference_answer", "")).strip()
            if not question or not answer:
                continue
            result = assistant.answer(
                question,
                game=game,
                player_state=record.get("player_state"),
            )
            if result.status not in {"found", "needs_context", "not_found", "ambiguous", "partial"}:
                continue
            output.append(
                build_grounded_sft_record(
                    example_id=str(record.get("id") or f"grounded_{index:06d}"),
                    result=result,
                    target_answer=answer,
                    split=str(record.get("split") or args.split),
                    category=record.get("category"),
                    required_facts=list(record.get("required_facts") or []),
                    forbidden_errors=list(
                        record.get("forbidden_errors")
                        or record.get("must_not_include")
                        or []
                    ),
                    prompt_mode=args.prompt_mode,
                    evidence_config=evidence_config,
                    target_source="reviewed_reference",
                )
            )
    write_grounded_sft(output, args.output)
    print(f"Wrote {len(output)} evidence-conditioned SFT examples to {args.output}")


if __name__ == "__main__":
    main()
