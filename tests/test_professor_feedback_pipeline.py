from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/run_professor_feedback_evaluation.py"
SPEC = importlib.util.spec_from_file_location("professor_feedback_pipeline", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_combine_with_deterministic_baseline_matches_frozen_ids(tmp_path: Path) -> None:
    quality = tmp_path / "quality.jsonl"
    deterministic = tmp_path / "deterministic.jsonl"
    output = tmp_path / "combined.jsonl"

    _write_jsonl(
        quality,
        [
            {"id": "a", "condition": "ungrounded_target", "output": "A0"},
            {"id": "a", "condition": "grounded_target", "output": "A1"},
            {"id": "b", "condition": "ungrounded_target", "output": "B0"},
            {"id": "b", "condition": "grounded_target", "output": "B1"},
        ],
    )
    _write_jsonl(
        deterministic,
        [
            {"annotation": {"id": "a"}, "result": {"answer": "DA"}},
            {"annotation": {"id": "b"}, "result": {"answer": "DB"}},
            {"annotation": {"id": "c"}, "result": {"answer": "DC"}},
        ],
    )

    summary = MODULE.combine_with_deterministic_baseline(
        quality,
        deterministic,
        output,
    )
    rows = MODULE.load_prediction_rows(output)

    assert summary == {
        "frozen_quality_rows": 4,
        "matched_evaluation_ids": 2,
        "deterministic_baseline_rows": 2,
        "combined_rows": 6,
    }
    assert [
        row for row in rows if row["condition"] == "deterministic_evidence_renderer"
    ] == [
        {"id": "a", "condition": "deterministic_evidence_renderer", "output": "DA"},
        {"id": "b", "condition": "deterministic_evidence_renderer", "output": "DB"},
    ]


def test_load_prediction_rows_supports_csv(tmp_path: Path) -> None:
    path = tmp_path / "quality.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "condition", "output"])
        writer.writeheader()
        writer.writerow({"id": "a", "condition": "grounded", "output": "answer"})

    assert MODULE.load_prediction_rows(path) == [
        {"id": "a", "condition": "grounded", "output": "answer"}
    ]
