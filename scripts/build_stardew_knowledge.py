#!/usr/bin/env python3
"""Build the compact Stardew Valley structured-fact database."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.games.stardew import build_stardew_database


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts", type=Path)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    kwargs = {}
    if args.facts:
        kwargs["facts_path"] = args.facts
    if args.database:
        kwargs["database_path"] = args.database
    if args.report:
        kwargs["report_path"] = args.report
    report = build_stardew_database(**kwargs)
    if not args.quiet:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
