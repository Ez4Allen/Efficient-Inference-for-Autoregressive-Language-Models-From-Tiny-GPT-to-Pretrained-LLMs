from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.clean_stardew_sft_data import _assign_groups, clean_records, run


def _record(identifier: str, question: str, source: str) -> dict:
    return {
        "id": identifier,
        "split": "train",
        "domain": "stardew_valley",
        "category": "fish_location",
        "language": "en",
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "A sourced answer."},
        ],
        "required_facts": ["sourced"],
        "forbidden_errors": ["unsupported"],
        "source_urls": [source],
        "verified": True,
    }


def test_cleanup_resets_unsubstantiated_review_state() -> None:
    accepted, rejected, report = clean_records([
        _record("x", "Where can I catch Catfish?", "https://stardewvalleywiki.com/Catfish#Locations")
    ])
    assert not rejected
    row = accepted[0].record
    assert row["verified"] is False
    assert row["review_status"] == "pending"
    assert row["reviewed_by"] is None
    assert row["source_urls"] == ["https://stardewvalleywiki.com/Catfish"]
    assert report["accepted_before_split"] == 1


def test_split_fraction_validation() -> None:
    with pytest.raises(ValueError, match="less than 1"):
        _assign_groups([], val_fraction=0.6, eval_fraction=0.4, seed=1)
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        _assign_groups([], val_fraction=-0.1, eval_fraction=0.1, seed=1)


def test_small_dataset_is_split_without_negative_counts(tmp_path: Path) -> None:
    source = tmp_path / "candidates.jsonl"
    source.write_text(json.dumps(_record("x", "Where can I catch Catfish?", "https://stardewvalleywiki.com/Catfish")) + "\n", encoding="utf-8")
    out = tmp_path / "out"
    report = run(source, out, val_fraction=0.1, eval_fraction=0.1, seed=42)
    assert sum(report["split_counts"].values()) == 1
    assert min(report["split_counts"].values()) >= 0
    assert sum(1 for name in ("train", "validation", "eval") if (out / f"{name}.jsonl").read_text(encoding="utf-8")) == 1


def test_source_and_template_groups_do_not_cross_splits(tmp_path: Path) -> None:
    records = [
        _record("a", "Where can I catch Catfish?", "https://stardewvalleywiki.com/Catfish"),
        _record("b", "When can I catch Catfish?", "https://stardewvalleywiki.com/Catfish"),
        _record("c", "Where can I catch Eel?", "https://stardewvalleywiki.com/Eel"),
        _record("d", "How do I craft Keg?", "https://stardewvalleywiki.com/Keg"),
    ]
    source = tmp_path / "candidates.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    report = run(source, tmp_path / "out", val_fraction=0.25, eval_fraction=0.25, seed=7)
    assert all(value == 0 for value in report["cross_split_source_overlap"].values())
    assert all(value == 0 for value in report["cross_split_template_overlap"].values())
