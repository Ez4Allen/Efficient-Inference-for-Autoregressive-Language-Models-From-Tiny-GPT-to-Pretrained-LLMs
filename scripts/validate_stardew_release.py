#!/usr/bin/env python3
"""Validate the complete Stardew Valley course-release data contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.games.stardew.database_builder import DEFAULT_DATABASE_PATH, validate_record
from src.utils.io import read_jsonl, write_json

CATALOG_ROOT = PROJECT_ROOT / "data" / "stardew" / "catalog"
EVALUATION_ROOT = PROJECT_ROOT / "data" / "stardew" / "evaluation"
TRAINING_ROOT = PROJECT_ROOT / "data" / "stardew" / "training"
GUIDES_ROOT = PROJECT_ROOT / "data" / "stardew" / "guides"
SFT_ROOT = PROJECT_ROOT / "data" / "stardew" / "sft"

EXPECTED_RECORD_MINIMUMS = {
    "crop": 30,
    "fish": 50,
    "villager": 20,
    "recipe": 100,
    "bundle": 30,
    "acquisition": 150,
}
EXPECTED_EVAL = {
    "validation_count": 40,
    "eval_count": 60,
    "language_distribution": {"en": 50, "zh": 50},
    "category_distribution": {
        "crop": 20,
        "fish": 15,
        "villager": 15,
        "recipe": 15,
        "bundle": 15,
        "acquisition": 10,
        "guide": 10,
    },
    "status_distribution": {"found": 70, "needs_context": 10, "partial": 10, "not_found": 10},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_question(value: str) -> str:
    import re
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(value).casefold())


def validate(*, strict_human_review: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}

    facts_path = CATALOG_ROOT / "cleaned" / "facts.jsonl"
    catalog_manifest_path = CATALOG_ROOT / "snapshot_manifest.json"
    facts = read_jsonl(facts_path)
    ids: set[str] = set()
    record_counts = Counter()
    for row in facts:
        try:
            validate_record(row)
        except Exception as exc:  # noqa: BLE001 - aggregate all release errors
            errors.append(f"catalog_schema:{row.get('source_catalog_id')}: {exc}")
        identifier = str(row.get("source_catalog_id"))
        if identifier in ids:
            errors.append(f"duplicate_catalog_id:{identifier}")
        ids.add(identifier)
        record_counts[str(row.get("record_type"))] += 1
    manifest = json.loads(catalog_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("record_count") != len(facts):
        errors.append("catalog_manifest_record_count_mismatch")
    if manifest.get("record_type_counts") != dict(sorted(record_counts.items())):
        errors.append("catalog_manifest_type_counts_mismatch")
    if manifest.get("facts_sha256") != sha256(facts_path):
        errors.append("catalog_manifest_sha256_mismatch")
    for record_type, minimum in EXPECTED_RECORD_MINIMUMS.items():
        if record_counts[record_type] < minimum:
            errors.append(f"catalog_minimum_not_met:{record_type}:{record_counts[record_type]}<{minimum}")
    standard_bundles = [row for row in facts if row.get("record_type") == "bundle" and (row.get("facts") or {}).get("bundle_mode") == "standard"]
    if len(standard_bundles) != 30:
        errors.append(f"standard_bundle_count:{len(standard_bundles)}!=30")
    acquisition_relations = sum(len((row.get("facts") or {}).get("sources") or []) for row in facts if row.get("record_type") == "acquisition")
    if acquisition_relations < 150:
        errors.append(f"acquisition_relation_minimum:{acquisition_relations}<150")
    checks["catalog"] = {
        "records": len(facts),
        "record_type_counts": dict(sorted(record_counts.items())),
        "standard_bundles": len(standard_bundles),
        "acquisition_relations": acquisition_relations,
        "unique_ids": len(ids),
    }

    validation_path = EVALUATION_ROOT / "stardew_validation_v1.jsonl"
    eval_path = EVALUATION_ROOT / "stardew_eval_v1.jsonl"
    eval_manifest_path = EVALUATION_ROOT / "manifest_v1.json"
    validation_rows = read_jsonl(validation_path)
    eval_rows = read_jsonl(eval_path)
    eval_rows_all = [*validation_rows, *eval_rows]
    eval_ids = [str(row.get("id")) for row in eval_rows_all]
    if len(eval_ids) != len(set(eval_ids)):
        errors.append("duplicate_evaluation_ids")
    for row in validation_rows:
        if row.get("split") != "validation" or "validation" not in str(row.get("id")):
            errors.append(f"validation_split_id_mismatch:{row.get('id')}")
    for row in eval_rows:
        if row.get("split") != "eval" or "eval" not in str(row.get("id")):
            errors.append(f"eval_split_id_mismatch:{row.get('id')}")
    language_counts = Counter(str(row.get("language")) for row in eval_rows_all)
    category_counts = Counter(str(row.get("category")) for row in eval_rows_all)
    status_counts = Counter(str(row.get("expected_status")) for row in eval_rows_all)
    if len(validation_rows) != EXPECTED_EVAL["validation_count"]:
        errors.append(f"validation_count:{len(validation_rows)}!=40")
    if len(eval_rows) != EXPECTED_EVAL["eval_count"]:
        errors.append(f"eval_count:{len(eval_rows)}!=60")
    if dict(language_counts) != EXPECTED_EVAL["language_distribution"]:
        errors.append(f"language_distribution:{dict(language_counts)}")
    if dict(category_counts) != EXPECTED_EVAL["category_distribution"]:
        errors.append(f"category_distribution:{dict(category_counts)}")
    if dict(status_counts) != EXPECTED_EVAL["status_distribution"]:
        errors.append(f"status_distribution:{dict(status_counts)}")
    for row in eval_rows_all:
        review_status = row.get("review_status")
        reviewer = row.get("reviewer")
        if strict_human_review:
            if review_status != "approved" or not reviewer or reviewer == row.get("annotator"):
                errors.append(f"human_review_not_approved:{row.get('id')}")
        elif review_status != "machine_validated" or not row.get("human_review_required"):
            errors.append(f"review_state_not_honest:{row.get('id')}")
        for source in row.get("required_sources") or []:
            source_id = source.get("source_catalog_id")
            if source_id and source_id not in ids:
                errors.append(f"evaluation_source_missing:{row.get('id')}:{source_id}")
    eval_manifest = json.loads(eval_manifest_path.read_text(encoding="utf-8"))
    if eval_manifest.get("validation_sha256") != sha256(validation_path):
        errors.append("evaluation_validation_sha256_mismatch")
    if eval_manifest.get("eval_sha256") != sha256(eval_path):
        errors.append("evaluation_eval_sha256_mismatch")
    checks["evaluation"] = {
        "validation": len(validation_rows),
        "eval": len(eval_rows),
        "language_distribution": dict(language_counts),
        "category_distribution": dict(category_counts),
        "status_distribution": dict(status_counts),
        "review_status_distribution": dict(Counter(str(row.get("review_status")) for row in eval_rows_all)),
        "human_review_required": not strict_human_review,
    }

    train_path = TRAINING_ROOT / "stardew_grounded_train_v1.jsonl"
    train_validation_path = TRAINING_ROOT / "stardew_grounded_validation_v1.jsonl"
    training_manifest_path = TRAINING_ROOT / "manifest_v1.json"
    train_rows = read_jsonl(train_path)
    training_validation_rows = read_jsonl(train_validation_path)
    training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    if training_manifest.get("train_sha256") != sha256(train_path):
        errors.append("training_train_sha256_mismatch")
    if training_manifest.get("validation_sha256") != sha256(train_validation_path):
        errors.append("training_validation_sha256_mismatch")
    formal_questions = {normalize_question(row["question"]) for row in eval_rows_all}
    training_questions = {
        normalize_question(message["content"])
        for row in [*train_rows, *training_validation_rows]
        for message in row.get("messages") or []
        if message.get("role") == "user"
    }
    overlap = formal_questions & training_questions
    if overlap:
        errors.append(f"formal_evaluation_training_question_overlap:{len(overlap)}")
    if not training_manifest.get("formal_evaluation_files_excluded"):
        errors.append("training_manifest_does_not_assert_eval_exclusion")
    checks["training"] = {
        "train": len(train_rows),
        "validation": len(training_validation_rows),
        "formal_eval_question_overlap": len(overlap),
    }

    seed_path = GUIDES_ROOT / "seed" / "pages.jsonl"
    seed_rows = read_jsonl(seed_path)
    guide_titles = [str(row.get("title")) for row in seed_rows]
    if len(seed_rows) < 20:
        errors.append(f"guide_seed_page_minimum:{len(seed_rows)}<20")
    if len(guide_titles) != len(set(guide_titles)):
        errors.append("duplicate_guide_seed_titles")
    for row in seed_rows:
        html = str(row.get("html") or "")
        if not html:
            errors.append(f"empty_guide_seed:{row.get('title')}")
        elif row.get("content_sha256") != hashlib.sha256(html.encode("utf-8")).hexdigest():
            errors.append(f"guide_seed_hash_mismatch:{row.get('title')}")
        if "project_authored_summary" not in (row.get("quality_flags") or []):
            errors.append(f"guide_seed_provenance_flag_missing:{row.get('title')}")
    checks["guides"] = {"seed_pages": len(seed_rows), "unique_titles": len(set(guide_titles))}

    audit_path = SFT_ROOT / "audit_report.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        for overlap_map_name in ("cross_split_source_overlap", "cross_split_template_overlap"):
            if any(int(value) != 0 for value in (audit.get(overlap_map_name) or {}).values()):
                errors.append(f"legacy_sft_{overlap_map_name}_nonzero")
        if (audit.get("verified_distribution") or {}) != {"false": audit.get("accepted_before_split")}:
            errors.append("legacy_sft_verified_state_invalid")
        checks["legacy_sft_candidates"] = {
            "accepted": audit.get("accepted_before_split"),
            "rejected": audit.get("rejected_records"),
            "split_counts": audit.get("split_counts"),
            "review_status_distribution": audit.get("review_status_distribution"),
            "source_overlap": audit.get("cross_split_source_overlap"),
            "template_overlap": audit.get("cross_split_template_overlap"),
        }
    else:
        warnings.append("legacy_sft_audit_report_missing")

    database_path = DEFAULT_DATABASE_PATH
    if database_path.exists():
        connection = sqlite3.connect(database_path)
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        db_records = connection.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        acquisition_rows = connection.execute(
            "SELECT record_json FROM records WHERE record_type = 'acquisition'"
        ).fetchall()
        relation_count = sum(
            len((json.loads(row[0]).get("facts") or {}).get("sources") or [])
            for row in acquisition_rows
        )
        connection.close()
        if integrity != "ok":
            errors.append(f"database_integrity:{integrity}")
        if db_records != len(facts):
            errors.append(f"database_record_count:{db_records}!={len(facts)}")
        if relation_count != acquisition_relations:
            errors.append(f"database_acquisition_relation_count:{relation_count}!={acquisition_relations}")
        checks["database"] = {"integrity": integrity, "records": db_records, "acquisition_relations": relation_count}
    else:
        warnings.append("catalog_database_not_built")

    status = "passed" if not errors else "failed"
    release_readiness = (
        "engineering_passed_human_review_pending"
        if status == "passed" and not strict_human_review
        else ("fully_approved" if status == "passed" else "blocked")
    )
    return {
        "status": status,
        "release_readiness": release_readiness,
        "strict_human_review": strict_human_review,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-human-review", action="store_true")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "stardew" / "release_validation.json")
    args = parser.parse_args()
    result = validate(strict_human_review=args.strict_human_review)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
