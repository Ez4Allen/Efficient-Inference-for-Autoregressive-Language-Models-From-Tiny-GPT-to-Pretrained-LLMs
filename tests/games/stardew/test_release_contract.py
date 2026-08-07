from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from scripts.validate_stardew_release import EXPECTED_EVAL, validate
from src.utils.io import read_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_stardew_release_contract_passes() -> None:
    report = validate(strict_human_review=False)
    assert report["status"] == "passed"
    assert report["release_readiness"] == "engineering_passed_human_review_pending"
    assert not report["errors"]


def test_formal_evaluation_distribution_and_review_state() -> None:
    root = PROJECT_ROOT / "data" / "stardew" / "evaluation"
    rows = [
        *read_jsonl(root / "stardew_validation_v1.jsonl"),
        *read_jsonl(root / "stardew_eval_v1.jsonl"),
    ]
    assert len(rows) == 100
    assert Counter(row["language"] for row in rows) == Counter(EXPECTED_EVAL["language_distribution"])
    assert Counter(row["category"] for row in rows) == Counter(EXPECTED_EVAL["category_distribution"])
    assert Counter(row["expected_status"] for row in rows) == Counter(EXPECTED_EVAL["status_distribution"])
    assert {row["review_status"] for row in rows} == {"machine_validated"}
    assert all(row["human_review_required"] for row in rows)
    assert all(row["reviewer"] is None for row in rows)


def test_legacy_candidate_audit_has_no_cross_split_leakage() -> None:
    audit = json.loads(
        (PROJECT_ROOT / "data" / "stardew" / "sft" / "audit_report.json").read_text(encoding="utf-8")
    )
    assert audit["status"] == "passed"
    assert audit["accepted_before_split"] == 1262
    assert audit["review_status_distribution"] == {"pending": 1262}
    assert audit["verified_distribution"] == {"false": 1262}
    assert all(value == 0 for value in audit["cross_split_source_overlap"].values())
    assert all(value == 0 for value in audit["cross_split_template_overlap"].values())
