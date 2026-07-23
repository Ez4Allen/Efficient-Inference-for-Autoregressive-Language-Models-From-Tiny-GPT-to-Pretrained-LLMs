#!/usr/bin/env python3
"""Build the Terraria linked catalog and SQLite query database."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.knowledge.pipeline import build_terraria_knowledge
from src.utils.paths import TERRARIA_CATALOG_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Link the tracked cleaned Terraria snapshot, audit references, "
            "and create the SQLite query database."
        )
    )
    parser.add_argument(
        "--catalog-root",
        type=Path,
        default=TERRARIA_CATALOG_ROOT,
        help="Catalog root containing cleaned/, linked/, and reports.",
    )
    parser.add_argument(
        "--database-path",
        type=Path,
        default=None,
        help="Optional SQLite output path (defaults inside catalog root).",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional pipeline report path.",
    )
    parser.add_argument(
        "--no-strict-snapshot",
        action="store_true",
        help=(
            "Skip exact snapshot hash/count validation. Referential and "
            "SQLite integrity checks still run."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-stage JSON output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_terraria_knowledge(
        catalog_root=args.catalog_root,
        database_path=args.database_path,
        pipeline_report_path=args.report_path,
        strict_snapshot=not args.no_strict_snapshot,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
