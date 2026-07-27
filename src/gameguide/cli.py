
"""Command-line interface for the multi-game grounded assistant."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence

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
from src.utils.paths import PROJECT_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-game evidence-grounded Qwen assistant."
    )
    parser.add_argument("question", nargs="*")
    parser.add_argument(
        "--game",
        choices=("terraria", "stardew", "stardew_valley"),
        required=True,
        help="Select the game knowledge plug-in.",
    )
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--llm",
        action="store_true",
        help="Use evidence-grounded Qwen generation after retrieval.",
    )
    model_group.add_argument(
        "--ungrounded",
        action="store_true",
        help="Use Qwen without retrieved evidence as an ablation baseline.",
    )
    parser.add_argument(
        "--engine",
        choices=("target", "draft", "speculative"),
        default="target",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "gameguidelm_qwen3_pair.yaml",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("evidence_only", "scaffolded"),
        help="Model-input ablation. evidence_only is the deployment default.",
    )
    parser.add_argument(
        "--evidence-policy",
        choices=("compact", "full", "structured_only", "guide_only"),
        help="Select which evidence is exposed to the language model.",
    )
    parser.add_argument("--max-evidence-sources", type=int)
    parser.add_argument("--max-evidence-characters", type=int)
    parser.add_argument("--language", choices=("auto", "en", "zh"), default="auto")
    parser.add_argument("--season")
    parser.add_argument("--day", type=int)
    parser.add_argument("--weather")
    parser.add_argument("--time")
    parser.add_argument("--location")
    parser.add_argument("--mode", choices=("normal", "expert", "master"), default="normal")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def create_plugin(game: str):
    if game == "terraria":
        return TerrariaGamePlugin(auto_build=True)
    return StardewAssistant(auto_build=True)


def _player_state(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: value
        for key, value in {
            "season": args.season,
            "day": args.day,
            "weather": args.weather,
            "time": args.time,
            "location": args.location,
            "mode": args.mode,
        }.items()
        if value is not None
    }


def emit(assistant: GameGuideAssistant, question: str, args: argparse.Namespace) -> None:
    result = assistant.answer(
        question,
        game=args.game,
        language=args.language,
        player_state=_player_state(args),
        include_debug=args.debug,
    )
    if args.json:
        print(json.dumps(result.to_dict(include_debug=args.debug), ensure_ascii=False, indent=2))
        return

    print(result.answer)
    cited = set(re.findall(r"\[(S\d+)\]", result.answer))
    if cited:
        sources = [item for item in result.evidence if item.source_id in cited]
    else:
        # Deterministic/fallback answers may not contain citation labels. Still
        # expose their provenance rather than silently hiding available sources.
        sources = list(result.evidence[:6])
    if sources:
        print("\nSources:")
        for item in sources:
            location = item.source_url or item.source_catalog_id
            print(f"- [{item.source_id}] {item.label}: {location}")
    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"- {warning}")
    if args.debug:
        print("\nDebug:")
        print(json.dumps(result.debug, ensure_ascii=False, indent=2))


def _create_generator(args: argparse.Namespace):
    if not (args.llm or args.ungrounded):
        return None
    config = load_qwen_pair_config(args.config)
    runtime = QwenPairRuntime(config)
    if args.ungrounded:
        return UngroundedQwenGenerator(
            runtime,
            engine=args.engine,
            fallback_on_error=config.grounding.fallback_on_error,
        )

    prompt_mode = args.prompt_mode or config.grounding.prompt_mode
    evidence_policy = args.evidence_policy or config.grounding.evidence_policy
    max_sources = args.max_evidence_sources or config.grounding.max_evidence_sources
    max_characters = (
        args.max_evidence_characters or config.grounding.max_evidence_characters
    )
    evidence_config = EvidenceSelectionConfig(
        policy=evidence_policy,
        max_sources=max_sources,
        max_characters=max_characters,
    )
    return GameGuideQwenGenerator(
        runtime,
        engine=args.engine,
        require_citations=config.grounding.require_citations,
        fallback_on_error=config.grounding.fallback_on_error,
        max_answer_chars=config.grounding.max_answer_chars,
        prompt_mode=prompt_mode,
        evidence_config=evidence_config,
        max_repair_attempts=config.grounding.max_repair_attempts,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    plugin = create_plugin(args.game)
    generator = _create_generator(args)
    with GameGuideAssistant([plugin], generator=generator) as assistant:
        if args.question:
            emit(assistant, " ".join(args.question), args)
            return
        print(f"GameGuideLM ({args.game}). Type 'exit' to stop.")
        while True:
            try:
                question = input("gameguide> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if question.casefold() in {"exit", "quit", "/exit", "/quit"}:
                break
            if question:
                emit(assistant, question, args)
                print()


if __name__ == "__main__":
    main()
