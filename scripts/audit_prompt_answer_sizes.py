#!/usr/bin/env python3
"""Report min/median/p95/max prompt and answer sizes from experiment rows."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.size_audit import audit_size_rows
from src.utils.io import read_jsonl, write_json


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.casefold() == ".jsonl":
        return read_jsonl(path)
    if path.suffix.casefold() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError("Input must be JSONL or CSV.")


def _flatten_runtime(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened = []
    for row in rows:
        candidate = dict(row)
        runtime = row.get("runtime")
        if isinstance(runtime, dict):
            for key in ("prompt_tokens", "generated_tokens"):
                if candidate.get(key) is None:
                    candidate[key] = runtime.get(key)
        flattened.append(candidate)
    return flattened


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-field", default="condition")
    parser.add_argument("--prompt-limit", type=int)
    parser.add_argument("--answer-limit", type=int)
    parser.add_argument("--evidence-source-limit", type=int)
    parser.add_argument("--evidence-character-limit", type=int)
    args = parser.parse_args()

    rows = _flatten_runtime(_load_rows(args.input))
    report = {
        "input": str(args.input.resolve()),
        "rows": len(rows),
        "limits": {
            "prompt_tokens": args.prompt_limit,
            "answer_tokens": args.answer_limit,
            "evidence_sources": args.evidence_source_limit,
            "evidence_characters": args.evidence_character_limit,
        },
        "groups": audit_size_rows(rows, group_field=args.group_field),
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
