#!/usr/bin/env python3
"""Build the Stardew Valley Wiki guide corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.games.stardew import build_stardew_guides, build_stardew_seed_guides


def main() -> None:
    parser = argparse.ArgumentParser()
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--offline", action="store_true")
    source_group.add_argument(
        "--seed",
        action="store_true",
        help="Build the tracked offline demonstration corpus instead of downloading Wiki pages.",
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.seed:
        if args.refresh or args.max_pages is not None:
            parser.error("--seed cannot be combined with --refresh or --max-pages")
        report = build_stardew_seed_guides(verbose=not args.quiet)
    else:
        report = build_stardew_guides(
            offline=args.offline,
            refresh=args.refresh,
            max_pages=args.max_pages,
            verbose=not args.quiet,
        )
    if not args.quiet:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
