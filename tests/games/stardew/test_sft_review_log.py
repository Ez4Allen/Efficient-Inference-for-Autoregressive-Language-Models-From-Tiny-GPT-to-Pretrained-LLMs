from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.clean_stardew_sft_data as clean_mod


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _base_record(**overrides) -> dict:
    record = {
        "id": "legacy_0001",
        "category": "crop_growth",
        "language": "en",
        "messages": [
            {"role": "user", "content": "How long does a Parsnip take to grow?"},
            {"role": "assistant", "content": "A Parsnip takes 4 days to grow."},
        ],
        "required_facts": ["4 days"],
        "forbidden_errors": ["wrong number of days"],
        "source_urls": ["https://stardewvalleywiki.com/Parsnip"],
        "verified": True,
        "reviewed_by": "maggie",
        "generation_method": "ai_assisted",
        "dataset_version": "1.0",
    }
    record.update(overrides)
    return record


def _run_with_review_log(
    tmp_path: Path, records: list[dict], log_entries: list[dict]
) -> tuple[list[dict], clean_mod.CleaningStats]:
    input_path = tmp_path / "candidates.jsonl"
    _write_jsonl(input_path, records)
    log_path = tmp_path / "review_log.jsonl"
    _write_jsonl(log_path, log_entries)

    raw = clean_mod.load_raw_records(input_path)
    stats = clean_mod.CleaningStats()
    normalized = clean_mod.normalize_records(raw, stats)
    clean_mod.apply_structural_and_duplicate_checks(normalized, stats)
    review_log = clean_mod.load_review_log(log_path)
    clean_mod.apply_review_log(normalized, review_log, stats)
    clean_mod.assign_knowledge_groups(normalized, stats)

    return normalized, stats


def _review_entry(**overrides) -> dict:
    entry = {
        "legacy_id": "legacy_0001",
        "decision": "approved",
        "reviewer": "maggie",
        "reviewed_at": "2026-08-05",
        "review_notes": "Checked against the wiki table.",
    }
    entry.update(overrides)
    return entry


def test_approval_sets_verified_true_and_attribution(tmp_path: Path) -> None:
    normalized, _stats = _run_with_review_log(
        tmp_path, [_base_record()], [_review_entry(decision="approved")]
    )
    record = normalized[0]
    assert record["review_status"] == "approved"
    assert record["verified"] is True
    assert record["reviewer"] == "maggie"
    assert record["reviewed_at"] == "2026-08-05"
    assert record["review_notes"] == "Checked against the wiki table."


def test_rejection_sets_verified_false_and_reasons(tmp_path: Path) -> None:
    normalized, _stats = _run_with_review_log(
        tmp_path,
        [_base_record()],
        [_review_entry(decision="rejected", rejection_reasons=["factual_review_required"])],
    )
    record = normalized[0]
    assert record["review_status"] == "rejected"
    assert record["verified"] is False
    assert record["rejection_reasons"] == ["factual_review_required"]


def test_needs_changes_decision(tmp_path: Path) -> None:
    normalized, _stats = _run_with_review_log(
        tmp_path, [_base_record()], [_review_entry(decision="needs_changes")]
    )
    record = normalized[0]
    assert record["review_status"] == "needs_changes"
    assert record["verified"] is False


def test_human_approval_overrides_automated_rejection(tmp_path: Path) -> None:
    dup_answer = "A Parsnip takes 4 days to grow."
    records = [
        _base_record(
            id="first",
            messages=[
                {"role": "user", "content": "How long does a Parsnip take to grow?"},
                {"role": "assistant", "content": dup_answer},
            ],
        ),
        _base_record(
            id="second",
            messages=[
                {"role": "user", "content": "How many days for a Parsnip to mature?"},
                {"role": "assistant", "content": dup_answer},
            ],
        ),
    ]
    # The automated pass will reject "second" as duplicate_answer; a human
    # reviewer determines it's actually fine (e.g. distinct enough in
    # context) and approves it anyway.
    normalized, stats = _run_with_review_log(
        tmp_path, records, [_review_entry(legacy_id="second", decision="approved")]
    )
    second = next(r for r in normalized if r["legacy_id"] == "second")
    assert second["review_status"] == "approved"
    assert second["verified"] is True
    assert stats.review_log_overrides_of_automated_rejection == 1


def test_human_rejection_overrides_automated_pass(tmp_path: Path) -> None:
    normalized, stats = _run_with_review_log(
        tmp_path, [_base_record()], [_review_entry(decision="rejected")]
    )
    record = normalized[0]
    assert record["review_status"] == "rejected"
    assert stats.review_log_overrides_to_rejection == 1


def test_unmatched_review_log_entries_are_reported(tmp_path: Path) -> None:
    _normalized, stats = _run_with_review_log(
        tmp_path,
        [_base_record()],
        [_review_entry(legacy_id="does_not_exist")],
    )
    assert stats.review_log_unmatched_ids == ["does_not_exist"]
    assert stats.review_log_applied_count == 0


def test_review_log_entry_without_reviewer_is_rejected(tmp_path: Path) -> None:
    log_path = tmp_path / "review_log.jsonl"
    _write_jsonl(log_path, [_review_entry(reviewer="")])
    with pytest.raises(SystemExit):
        clean_mod.load_review_log(log_path)


def test_review_log_entry_with_invalid_decision_is_rejected(tmp_path: Path) -> None:
    log_path = tmp_path / "review_log.jsonl"
    _write_jsonl(log_path, [_review_entry(decision="accepted")])
    with pytest.raises(SystemExit):
        clean_mod.load_review_log(log_path)


def test_review_log_entry_without_reviewed_at_is_rejected(tmp_path: Path) -> None:
    log_path = tmp_path / "review_log.jsonl"
    _write_jsonl(log_path, [_review_entry(reviewed_at="")])
    with pytest.raises(SystemExit):
        clean_mod.load_review_log(log_path)


def test_review_log_entry_without_legacy_id_is_rejected(tmp_path: Path) -> None:
    log_path = tmp_path / "review_log.jsonl"
    entry = _review_entry()
    del entry["legacy_id"]
    _write_jsonl(log_path, [entry])
    with pytest.raises(SystemExit):
        clean_mod.load_review_log(log_path)


def test_pipeline_runs_without_a_review_log_file(tmp_path: Path) -> None:
    # main() treats a missing review log as "no overrides", not an error.
    normalized, stats = _run_with_review_log(tmp_path, [_base_record()], [])
    assert normalized[0]["review_status"] == "pending"
    assert stats.review_log_applied_count == 0
