#!/usr/bin/env python3
"""Render a worked, machine-readable explanation of one answer score."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.gameguide_eval import build_answer_validation_trace
from src.gameguide.schemas import GameEvidence, GameGuideResult
from src.utils.io import read_jsonl, write_json


def result_from_dict(payload: dict[str, Any]) -> GameGuideResult:
    evidence = [
        GameEvidence(**item)
        for item in payload.get("evidence") or []
        if isinstance(item, dict)
    ]
    return GameGuideResult(
        game=str(payload.get("game") or "unknown"),
        status=str(payload.get("status") or "unknown"),
        question=str(payload.get("question") or ""),
        intent=str(payload.get("intent") or "unknown"),
        entity=(str(payload["entity"]) if payload.get("entity") is not None else None),
        answer=str(payload.get("answer") or ""),
        facts=payload.get("facts") if isinstance(payload.get("facts"), dict) else None,
        warnings=[str(item) for item in payload.get("warnings") or []],
        candidates=list(payload.get("candidates") or []),
        evidence=evidence,
        context_payload=dict(payload.get("context_payload") or {}),
        debug=dict(payload.get("debug") or {}),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    selected = None
    for row in read_jsonl(args.input):
        annotation = row.get("annotation")
        if isinstance(annotation, dict) and str(annotation.get("id")) == args.id:
            selected = row
            break
    if selected is None:
        raise KeyError(f"Example {args.id!r} not found in {args.input}")
    annotation = selected.get("annotation")
    result_payload = selected.get("result")
    if not isinstance(annotation, dict) or not isinstance(result_payload, dict):
        raise TypeError("Input row must contain annotation and result objects.")

    trace = build_answer_validation_trace(
        annotation,
        result_from_dict(result_payload),
    ).to_dict()
    payload = {
        "annotation": annotation,
        "trace": trace,
        "interpretation": {
            "required_fact_coverage": (
                "Matched required facts divided by all required facts. Short facts "
                "need complete semantic-token recall; longer facts use a 0.75 recall threshold."
            ),
            "pass_formula": trace["pass_formula"],
            "runtime_validator": (
                "Separately checks source IDs, citations, evidence-supported numbers, "
                "answer length, and leaked thinking text before evaluation scoring."
            ),
        },
    }
    if args.output is not None:
        write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
