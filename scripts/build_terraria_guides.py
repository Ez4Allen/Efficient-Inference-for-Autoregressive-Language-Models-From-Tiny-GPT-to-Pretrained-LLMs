#!/usr/bin/env python3
"""Build the local Terraria guide corpus and SQLite FTS index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.pipeline import DEFAULT_GUIDES_ROOT, build_terraria_guides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover Official Terraria Wiki guide pages, import text, clean and "
            "chunk it, then build a local SQLite FTS retrieval database."
        )
    )
    parser.add_argument("--guides-root", type=Path, default=DEFAULT_GUIDES_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip network import and rebuild from raw/pages.jsonl.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refetch pages even when the stored revision ID is unchanged.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Limit discovered pages for a smoke test.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_terraria_guides(
        guides_root=args.guides_root,
        manifest_path=args.manifest,
        offline=args.offline,
        refresh=args.refresh,
        max_pages=args.max_pages,
        verbose=not args.quiet,
    )
    if not args.quiet:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
