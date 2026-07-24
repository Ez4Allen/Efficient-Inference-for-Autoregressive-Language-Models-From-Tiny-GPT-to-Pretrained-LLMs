#!/usr/bin/env python3
"""Run the grounded Terraria pipeline with paired Qwen inference."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.assistant import QwenGroundedAnswerGenerator, TerrariaAssistant
from src.inference.chat_runtime import QwenPairRuntime
from src.knowledge.terraria_query_store import DEFAULT_DATABASE_PATH
from src.models.runtime_config import load_qwen_pair_config
from src.retrieval.guide_database import DEFAULT_GUIDE_DATABASE_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grounded Terraria QA using a Qwen3 draft/target model pair."
    )
    parser.add_argument("question", nargs="*", help="Question; omit for interactive mode.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "terraria_qwen3_pair.yaml",
    )
    parser.add_argument("--engine", choices=("target", "draft", "speculative"))
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument(
        "--guides-database", type=Path, default=DEFAULT_GUIDE_DATABASE_PATH
    )
    parser.add_argument("--auto-build", action="store_true")
    parser.add_argument("--mode", choices=("normal", "expert", "master"), default="normal")
    parser.add_argument("--language", choices=("auto", "en", "zh"), default="auto")
    parser.add_argument("--guide-limit", type=int, default=6)
    parser.add_argument("--guide-minimum-score", type=float, default=0.14)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _print_cited_sources(response) -> None:
    cited = set(re.findall(r"\[(S\d+)\]", response.answer))
    if not cited:
        return
    sources = [
        item
        for item in response.evidence
        if str(item.get("source_id")) in cited
    ]
    if not sources:
        return
    print("\nSources:")
    for item in sources:
        source_id = item.get("source_id")
        title = item.get("page_title") or item.get("entity_type") or "evidence"
        section = item.get("section_title")
        label = f"{title} — {section}" if section else str(title)
        location = item.get("source_url") or item.get("source_catalog_id") or "local evidence"
        print(f"- [{source_id}] {label}: {location}")


def emit(assistant: TerrariaAssistant, question: str, args: argparse.Namespace) -> None:
    response = assistant.answer(
        question,
        mode=args.mode,
        language=args.language,
        guide_limit=args.guide_limit,
        guide_minimum_score=args.guide_minimum_score,
        include_debug=args.debug,
    )
    if args.json:
        print(
            json.dumps(
                response.to_dict(include_debug=args.debug),
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(response.answer)
        _print_cited_sources(response)
        if response.warnings:
            print("\nWarnings:")
            for warning in response.warnings:
                print(f"- {warning}")
        if args.debug and response.debug:
            print("\nDebug:")
            print(json.dumps(response.debug, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    config = load_qwen_pair_config(args.config)
    runtime = QwenPairRuntime(config)
    generator = QwenGroundedAnswerGenerator(
        runtime,
        engine=args.engine or config.generation.engine,
        require_citations=config.grounding.require_citations,
        fallback_on_error=config.grounding.fallback_on_error,
        max_answer_chars=config.grounding.max_answer_chars,
    )

    with TerrariaAssistant(
        args.database,
        guide_database_path=args.guides_database,
        auto_build=args.auto_build,
        generator=generator,
    ) as assistant:
        if args.question:
            emit(assistant, " ".join(args.question), args)
            return

        print("Grounded Terraria Qwen assistant. Type 'exit' or 'quit' to stop.")
        while True:
            try:
                question = input("terraria-llm> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not question:
                continue
            if question.casefold() in {"exit", "quit", "/exit", "/quit"}:
                break
            emit(assistant, question, args)
            print()


if __name__ == "__main__":
    main()
