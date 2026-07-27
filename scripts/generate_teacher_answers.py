
#!/usr/bin/env python3
"""Generate validated Qwen3-4B teacher answers for draft-model adaptation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gameguide import EvidenceSelectionConfig, GameGuideAssistant, GameGuideQwenGenerator
from src.games.stardew import StardewAssistant
from src.games.terraria import TerrariaGamePlugin
from src.inference.chat_runtime import QwenPairRuntime
from src.models.runtime_config import load_qwen_pair_config
from src.training.grounded_sft import (
    build_grounded_sft_record,
    load_annotation_records,
    write_grounded_sft,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "gameguidelm_qwen3_pair.yaml",
    )
    parser.add_argument("--default-game", choices=("terraria", "stardew_valley"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    config = load_qwen_pair_config(args.config)
    evidence_config = EvidenceSelectionConfig(
        policy=config.grounding.evidence_policy,
        max_sources=config.grounding.max_evidence_sources,
        max_characters=config.grounding.max_evidence_characters,
    )
    generator = GameGuideQwenGenerator(
        QwenPairRuntime(config),
        engine="target",
        require_citations=config.grounding.require_citations,
        fallback_on_error=config.grounding.fallback_on_error,
        max_answer_chars=config.grounding.max_answer_chars,
        prompt_mode=config.grounding.prompt_mode,
        evidence_config=evidence_config,
        max_repair_attempts=config.grounding.max_repair_attempts,
    )
    annotations = load_annotation_records(args.input)
    if args.limit is not None:
        annotations = annotations[: max(0, args.limit)]
    output = []
    with GameGuideAssistant(
        [TerrariaGamePlugin(auto_build=True), StardewAssistant(auto_build=True)],
        generator=generator,
    ) as assistant:
        for index, record in enumerate(annotations, start=1):
            game = record.get("game") or args.default_game
            if not game:
                raise ValueError(f"Record {index} has no game.")
            game = "stardew_valley" if game == "stardew" else game
            result = assistant.answer(
                str(record["question"]),
                game=game,
                player_state=record.get("player_state"),
                include_debug=True,
            )
            generation = result.debug.get("generation") or {}
            if generation.get("fallback_used") or result.status != "found":
                continue
            output.append(
                build_grounded_sft_record(
                    example_id=f"teacher_{record.get('id', index)}",
                    result=result,
                    target_answer=result.answer,
                    split="train",
                    category=record.get("category"),
                    required_facts=list(record.get("required_facts") or []),
                    forbidden_errors=list(record.get("forbidden_errors") or []),
                    prompt_mode=config.grounding.prompt_mode,
                    evidence_config=evidence_config,
                    target_source="validated_qwen3_4b_teacher",
                )
            )
            print(
                json.dumps(
                    {"index": index, "game": game, "question": record["question"]},
                    ensure_ascii=False,
                )
            )
    write_grounded_sft(output, args.output)
    print(f"Wrote {len(output)} validated target-teacher examples to {args.output}")


if __name__ == "__main__":
    main()
