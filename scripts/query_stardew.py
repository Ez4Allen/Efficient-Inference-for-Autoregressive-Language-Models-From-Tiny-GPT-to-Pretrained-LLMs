#!/usr/bin/env python3
"""Query the deterministic Stardew Valley fact service."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.games.stardew import StardewFactService, build_stardew_database
from src.games.stardew.database_builder import DEFAULT_DATABASE_PATH


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("intent", choices=sorted(StardewFactService.VALID_INTENTS))
    parser.add_argument("entity")
    parser.add_argument("--season")
    parser.add_argument("--day", type=int)
    parser.add_argument("--weather")
    parser.add_argument("--time")
    parser.add_argument("--location")
    parser.add_argument("--bundle-mode", default="standard")
    args = parser.parse_args()
    if not DEFAULT_DATABASE_PATH.exists():
        build_stardew_database()
    state = {
        key: value
        for key, value in {
            "season": args.season,
            "day": args.day,
            "weather": args.weather,
            "time": args.time,
            "location": args.location,
            "bundle_mode": args.bundle_mode,
        }.items()
        if value is not None
    }
    with StardewFactService() as service:
        result = service.query(
            args.intent,
            args.entity,
            player_state=state,
            bundle_mode=args.bundle_mode,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
