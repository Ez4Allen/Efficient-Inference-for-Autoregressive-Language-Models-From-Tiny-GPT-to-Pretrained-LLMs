from __future__ import annotations

import json
from pathlib import Path

import scripts.clean_stardew_sft_data as clean_mod


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _base_record(**overrides) -> dict:
    record = {
        "id": "legacy_0001",
        "split": "train",
        "domain": "stardew_valley",
        "category": "crop_growth",
        "language": "en",
        "messages": [
            {"role": "system", "content": "You are a reliable game guide assistant."},
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


def _run_pipeline(tmp_path: Path, records: list[dict]) -> tuple[list[dict], list[dict], dict]:
    input_path = tmp_path / "candidates.jsonl"
    _write_jsonl(input_path, records)

    outdir = tmp_path / "sft"
    report_path = tmp_path / "reports" / "cleaning_report.json"

    raw = clean_mod.load_raw_records(input_path)
    stats = clean_mod.CleaningStats()
    normalized = clean_mod.normalize_records(raw, stats)
    clean_mod.apply_structural_and_duplicate_checks(normalized, stats)
    clean_mod.assign_knowledge_groups(normalized, stats)

    clean_mod.write_jsonl(outdir / "candidates.normalized.jsonl", normalized)
    rejected = [r for r in normalized if r["review_status"] == "rejected"]
    clean_mod.write_jsonl(outdir / "rejected.jsonl", rejected)
    clean_mod.write_report(
        report_path, stats, len(normalized), len(normalized) - len(rejected), len(rejected)
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    return normalized, rejected, report


def test_stable_ids_are_assigned_and_legacy_id_preserved(tmp_path: Path) -> None:
    normalized, _rejected, _report = _run_pipeline(
        tmp_path, [_base_record(id="stardew_valley_train_x_0001")]
    )
    record = normalized[0]
    assert record["id"] == "sdv_sft_000001"
    assert record["legacy_id"] == "stardew_valley_train_x_0001"
    assert "train" not in record["id"]
    assert "validation" not in record["id"]
    assert "eval" not in record["id"]


def test_original_category_and_taxonomy_are_preserved(tmp_path: Path) -> None:
    normalized, _rejected, _report = _run_pipeline(
        tmp_path, [_base_record(category="crop_growth")]
    )
    record = normalized[0]
    assert record["original_category"] == "crop_growth"
    assert record["task_family"] == "crop_planning"


def test_unauditable_records_are_marked_pending_and_unverified(tmp_path: Path) -> None:
    normalized, _rejected, _report = _run_pipeline(
        tmp_path, [_base_record(verified=True, reviewed_by="maggie")]
    )
    record = normalized[0]
    # verified=True/reviewed_by=maggie in the input carries no auditable
    # evidence (no reviewer/reviewed_at/review_notes), so it must not be
    # trusted at face value.
    assert record["verified"] is False
    assert record["review_status"] == "pending"
    assert record["legacy_reviewed_by"] == "maggie"


def test_record_with_review_evidence_is_marked_approved(tmp_path: Path) -> None:
    normalized, _rejected, _report = _run_pipeline(
        tmp_path,
        [
            _base_record(
                reviewer="alice",
                reviewed_at="2026-01-01T00:00:00Z",
                review_notes="Checked against wiki table.",
            )
        ],
    )
    record = normalized[0]
    assert record["verified"] is True
    assert record["review_status"] == "approved"


def test_case_variant_sources_share_a_knowledge_group(tmp_path: Path) -> None:
    normalized, _rejected, _report = _run_pipeline(
        tmp_path,
        [
            _base_record(
                id="a",
                category="crop_growth",
                messages=[
                    {"role": "user", "content": "How long does a Parsnip take to grow?"},
                    {"role": "assistant", "content": "A Parsnip takes 4 days to grow."},
                ],
                source_urls=["https://stardewvalleywiki.com/Parsnip"],
            ),
            _base_record(
                id="b",
                category="crop_harvest",
                messages=[
                    {"role": "user", "content": "Does a Parsnip regrow after harvest?"},
                    {"role": "assistant", "content": "No, a Parsnip does not regrow."},
                ],
                source_urls=["https://stardewvalleywiki.com/parsnip"],
            ),
        ],
    )
    groups = {r["legacy_id"]: r["knowledge_group"] for r in normalized}
    assert groups["a"] == groups["b"]


def test_duplicate_answer_is_rejected_and_first_occurrence_kept(tmp_path: Path) -> None:
    dup_answer = "A Parsnip takes 4 days to grow."
    normalized, rejected, _report = _run_pipeline(
        tmp_path,
        [
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
        ],
    )
    assert len(rejected) == 1
    assert rejected[0]["legacy_id"] == "second"
    assert rejected[0]["rejection_reasons"] == ["duplicate_answer"]

    kept = [r for r in normalized if r["review_status"] != "rejected"]
    assert len(kept) == 1
    assert kept[0]["legacy_id"] == "first"


def test_duplicate_question_is_rejected(tmp_path: Path) -> None:
    question = "How long does a Parsnip take to grow?"
    normalized, rejected, _report = _run_pipeline(
        tmp_path,
        [
            _base_record(
                id="first",
                messages=[
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": "A Parsnip takes 4 days to grow."},
                ],
            ),
            _base_record(
                id="second",
                messages=[
                    {"role": "user", "content": question + "?"},
                    {"role": "assistant", "content": "It takes exactly four days."},
                ],
            ),
        ],
    )
    assert [r["rejection_reasons"] for r in rejected] == [["duplicate_question"]]


def test_malformed_messages_are_rejected(tmp_path: Path) -> None:
    _normalized, rejected, _report = _run_pipeline(
        tmp_path,
        [_base_record(messages=[{"role": "assistant", "content": "answer with no question"}])],
    )
    assert len(rejected) == 1
    assert "malformed_messages" in rejected[0]["rejection_reasons"]


def test_missing_source_is_rejected(tmp_path: Path) -> None:
    _normalized, rejected, _report = _run_pipeline(tmp_path, [_base_record(source_urls=[])])
    assert len(rejected) == 1
    assert "missing_source" in rejected[0]["rejection_reasons"]


def test_unsupported_source_is_rejected(tmp_path: Path) -> None:
    _normalized, rejected, _report = _run_pipeline(
        tmp_path, [_base_record(source_urls=["https://example.com/Parsnip"])]
    )
    assert len(rejected) == 1
    assert "unsupported_source" in rejected[0]["rejection_reasons"]


def test_unmapped_category_is_recorded_in_report(tmp_path: Path) -> None:
    _normalized, _rejected, report = _run_pipeline(
        tmp_path, [_base_record(category="totally_unrecognized_topic_xyz")]
    )
    assert report["unmapped_category_counts"] == {"totally_unrecognized_topic_xyz": 1}


def test_pipeline_never_touches_the_input_file(tmp_path: Path) -> None:
    input_path = tmp_path / "candidates.jsonl"
    _write_jsonl(input_path, [_base_record()])
    before = input_path.read_text(encoding="utf-8")

    raw = clean_mod.load_raw_records(input_path)
    stats = clean_mod.CleaningStats()
    normalized = clean_mod.normalize_records(raw, stats)
    clean_mod.apply_structural_and_duplicate_checks(normalized, stats)
    clean_mod.assign_knowledge_groups(normalized, stats)

    after = input_path.read_text(encoding="utf-8")
    assert before == after
