#!/usr/bin/env python3
"""Query the local Terraria guide FTS database."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.guide_database import (
    DEFAULT_GUIDE_DATABASE_PATH,
    GuideDocumentStore,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the local Terraria guide corpus.")
    parser.add_argument("query", nargs="+")
    parser.add_argument("--database", type=Path, default=DEFAULT_GUIDE_DATABASE_PATH)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--minimum-score", type=float, default=0.14)
    parser.add_argument("--exclude-low-quality", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    query = " ".join(args.query)
    with GuideDocumentStore(args.database) as store:
        hits = store.search(
            query,
            limit=args.limit,
            minimum_score=args.minimum_score,
            include_low_quality=not args.exclude_low_quality,
        )

    if args.json:
        print(json.dumps({"query": query, "hits": hits}, ensure_ascii=False, indent=2))
        return
    if not hits:
        print("No sufficiently relevant guide chunks were found.")
        return
    for hit in hits:
        print(f"[{hit['score']:.3f}] {hit['citation_label']}")
        print(hit["text"])
        print(hit["source_url"])
        print()


if __name__ == "__main__":
    main()
