"""Query the generated Terraria fact database from the command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.knowledge import TerrariaFactService  # noqa: E402
from src.knowledge.terraria_query_store import DEFAULT_DATABASE_PATH  # noqa: E402
from src.utils.paths import resolve_project_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "intent",
        choices=sorted(TerrariaFactService.VALID_INTENTS),
        help="Structured query intent.",
    )
    parser.add_argument("entity", help="Item, NPC, recipe result, or search text.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--mode", choices=["normal", "expert", "master"], default="normal")
    parser.add_argument("--npc-id", type=int, default=None)
    parser.add_argument("--item-id", type=int, default=None)
    parser.add_argument("--internal-name", default=None)
    parser.add_argument("--all-variants", action="store_true")
    parser.add_argument("--complete-only", action="store_true")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--compact", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    database_path = resolve_project_path(args.database)
    if not database_path.exists():
        raise FileNotFoundError(
            f"Terraria query database not found: {database_path}\n"
            "Build it with: python scripts/build_terraria_knowledge.py --quiet"
        )

    with TerrariaFactService(database_path) as service:
        result = service.query(
            args.intent,
            args.entity,
            mode=args.mode,
            npc_id=args.npc_id,
            item_id=args.item_id,
            internal_name=args.internal_name,
            preferred_only=not args.all_variants,
            include_partial=not args.complete_only,
            limit_per_type=args.limit,
        )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=None if args.compact else 2,
        )
    )


if __name__ == "__main__":
    main()
