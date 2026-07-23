#!/usr/bin/env python3
"""Command-line interface for the grounded Terraria assistant."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.assistant import TerrariaAssistant
from src.knowledge.terraria_query_store import DEFAULT_DATABASE_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ask natural-language Terraria questions grounded in the local "
            "structured knowledge database."
        )
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="Question to answer. Omit it to start an interactive session.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
        help="Path to terraria_query.sqlite3.",
    )
    parser.add_argument(
        "--auto-build",
        action="store_true",
        help="Build the query database from the tracked cleaned snapshot if missing.",
    )
    parser.add_argument(
        "--mode",
        choices=("normal", "expert", "master"),
        default="normal",
        help="Default game mode when the question does not specify one.",
    )
    parser.add_argument(
        "--language",
        choices=("auto", "en", "zh"),
        default="auto",
        help="Answer language. The default detects Chinese automatically.",
    )
    parser.add_argument(
        "--all-variants",
        action="store_true",
        help="Include legacy/non-preferred recipe variants.",
    )
    parser.add_argument(
        "--exclude-partial",
        action="store_true",
        help="Exclude Drop records containing unresolved legacy references.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete structured AssistantResponse as JSON.",
    )
    parser.add_argument(
        "--context",
        action="store_true",
        help="Print the grounded context bundle instead of the rendered answer.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Include route and latency diagnostics in JSON output.",
    )
    return parser.parse_args()


def emit(assistant: TerrariaAssistant, question: str, args: argparse.Namespace) -> None:
    response = assistant.answer(
        question,
        mode=args.mode,
        preferred_only=not args.all_variants,
        include_partial=not args.exclude_partial,
        language=args.language,
        include_debug=args.debug,
    )
    if args.context:
        print(response.context.text if response.context else "")
    elif args.json:
        print(
            json.dumps(
                response.to_dict(include_debug=args.debug),
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(response.answer)
        if response.warnings:
            print("\nWarnings:")
            for warning in response.warnings:
                print(f"- {warning}")


def main() -> None:
    args = parse_args()
    with TerrariaAssistant(
        args.database,
        auto_build=args.auto_build,
    ) as assistant:
        if args.question:
            emit(assistant, " ".join(args.question), args)
            return

        print("Grounded Terraria Assistant. Type 'exit' or 'quit' to stop.")
        while True:
            try:
                question = input("terraria> ").strip()
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
