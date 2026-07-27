
#!/usr/bin/env python3
"""Evaluate deterministic or Qwen-backed GameGuideLM on reviewed QA files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.gameguide_eval import evaluate_files
from src.gameguide import (
    EvidenceSelectionConfig,
    GameGuideAssistant,
    GameGuideQwenGenerator,
    UngroundedQwenGenerator,
)
from src.games.stardew import StardewAssistant
from src.games.terraria import TerrariaGamePlugin
from src.inference.chat_runtime import QwenPairRuntime
from src.models.runtime_config import load_qwen_pair_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--llm", action="store_true")
    model_group.add_argument("--ungrounded", action="store_true")
    parser.add_argument("--engine", choices=("target", "draft", "speculative"), default="target")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "gameguidelm_qwen3_pair.yaml",
    )
    parser.add_argument("--prompt-mode", choices=("evidence_only", "scaffolded"))
    parser.add_argument(
        "--evidence-policy",
        choices=("compact", "full", "structured_only", "guide_only"),
    )
    parser.add_argument(
        "--default-game",
        choices=("terraria", "stardew_valley"),
        help="Fallback for custom evaluation files that omit the game field.",
    )
    args = parser.parse_args()

    generator = None
    if args.llm or args.ungrounded:
        config = load_qwen_pair_config(args.config)
        runtime = QwenPairRuntime(config)
        if args.ungrounded:
            generator = UngroundedQwenGenerator(
                runtime,
                engine=args.engine,
                fallback_on_error=config.grounding.fallback_on_error,
            )
        else:
            evidence_config = EvidenceSelectionConfig(
                policy=args.evidence_policy or config.grounding.evidence_policy,
                max_sources=config.grounding.max_evidence_sources,
                max_characters=config.grounding.max_evidence_characters,
            )
            generator = GameGuideQwenGenerator(
                runtime,
                engine=args.engine,
                require_citations=config.grounding.require_citations,
                fallback_on_error=config.grounding.fallback_on_error,
                max_answer_chars=config.grounding.max_answer_chars,
                prompt_mode=args.prompt_mode or config.grounding.prompt_mode,
                evidence_config=evidence_config,
                max_repair_attempts=config.grounding.max_repair_attempts,
            )
    with GameGuideAssistant(
        [TerrariaGamePlugin(auto_build=True), StardewAssistant(auto_build=True)],
        generator=generator,
    ) as assistant:
        summary = evaluate_files(
            assistant,
            args.input,
            output_path=args.output,
            summary_path=args.summary,
            default_game=args.default_game,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
