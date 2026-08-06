from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.split_stardew_sft_data as split_mod


# --- compute_group_split_counts: tiny-dataset handling -----------------


def test_zero_groups_raises() -> None:
    with pytest.raises(SystemExit):
        split_mod.compute_group_split_counts(0, 0.1, 0.1)


def test_one_group_goes_entirely_to_train() -> None:
    assert split_mod.compute_group_split_counts(1, 0.1, 0.1) == (1, 0, 0)


def test_two_groups_with_both_fracs_splits_one_to_eval_on_tie() -> None:
    # eval_frac >= val_frac wins ties.
    assert split_mod.compute_group_split_counts(2, 0.1, 0.1) == (1, 0, 1)


def test_two_groups_val_only() -> None:
    assert split_mod.compute_group_split_counts(2, 0.5, 0.0) == (1, 1, 0)


def test_three_groups_with_both_fracs_gives_one_each() -> None:
    assert split_mod.compute_group_split_counts(3, 0.1, 0.1) == (1, 1, 1)


def test_three_groups_no_holdout_requested() -> None:
    assert split_mod.compute_group_split_counts(3, 0.0, 0.0) == (3, 0, 0)


def test_train_count_never_goes_negative_for_small_group_counts() -> None:
    for n in range(1, 10):
        n_train, n_val, n_eval = split_mod.compute_group_split_counts(n, 0.1, 0.1)
        assert n_train >= 1
        assert n_train + n_val + n_eval == n
        assert n_val >= 0
        assert n_eval >= 0


def test_general_case_rounds_and_stays_within_bounds() -> None:
    n_train, n_val, n_eval = split_mod.compute_group_split_counts(100, 0.1, 0.1)
    assert (n_train, n_val, n_eval) == (80, 10, 10)
    assert n_train + n_val + n_eval == 100


# --- validate_fractions --------------------------------------------------


@pytest.mark.parametrize(
    "val_frac,eval_frac",
    [(-0.1, 0.1), (0.1, -0.1), (1.0, 0.0), (0.0, 1.0), (0.6, 0.6)],
)
def test_invalid_fractions_are_rejected(val_frac: float, eval_frac: float) -> None:
    with pytest.raises(SystemExit):
        split_mod.validate_fractions(val_frac, eval_frac)


def test_valid_fractions_pass() -> None:
    split_mod.validate_fractions(0.1, 0.1)  # should not raise


# --- split_records: determinism and knowledge-group integrity -----------


def _record(record_id: str, knowledge_group: str, source_pages: list[str]) -> dict:
    return {
        "id": record_id,
        "knowledge_group": knowledge_group,
        "source_pages": source_pages,
        "messages": [
            {"role": "user", "content": f"Question about {record_id}?"},
            {"role": "assistant", "content": f"Answer for {record_id}."},
        ],
    }


def _fixture_records(n_groups: int, per_group: int = 2) -> list[dict]:
    records = []
    for g in range(n_groups):
        for i in range(per_group):
            records.append(
                _record(f"sdv_sft_{g:03d}_{i}", f"kg_{g:06d}", [f"Page_{g}"])
            )
    return records


def test_split_is_deterministic_for_fixed_seed_and_input() -> None:
    records = _fixture_records(20)
    train1, val1, eval1, counts1 = split_mod.split_records(records, 0.2, 0.2, seed=7)
    train2, val2, eval2, counts2 = split_mod.split_records(records, 0.2, 0.2, seed=7)

    ids1 = sorted(r["id"] for r in train1) + sorted(r["id"] for r in val1) + sorted(r["id"] for r in eval1)
    ids2 = sorted(r["id"] for r in train2) + sorted(r["id"] for r in val2) + sorted(r["id"] for r in eval2)
    assert ids1 == ids2
    assert counts1 == counts2


def test_different_seed_can_change_assignment() -> None:
    records = _fixture_records(20)
    _train1, val1, _eval1, _counts1 = split_mod.split_records(records, 0.2, 0.2, seed=1)
    _train2, val2, _eval2, _counts2 = split_mod.split_records(records, 0.2, 0.2, seed=2)
    val_ids_1 = {r["id"] for r in val1}
    val_ids_2 = {r["id"] for r in val2}
    assert val_ids_1 != val_ids_2


def test_knowledge_group_never_split_across_train_val_eval() -> None:
    records = _fixture_records(15, per_group=3)
    train, val, evalset, _counts = split_mod.split_records(records, 0.2, 0.2, seed=42)

    def groups_of(recs: list[dict]) -> set[str]:
        return {r["knowledge_group"] for r in recs}

    train_groups = groups_of(train)
    val_groups = groups_of(val)
    eval_groups = groups_of(evalset)

    assert not (train_groups & val_groups)
    assert not (train_groups & eval_groups)
    assert not (val_groups & eval_groups)


def test_split_assigns_split_field_consistently() -> None:
    records = _fixture_records(15, per_group=2)
    train, val, evalset, _counts = split_mod.split_records(records, 0.2, 0.2, seed=42)
    assert all(r["split"] == "train" for r in train)
    assert all(r["split"] == "validation" for r in val)
    assert all(r["split"] == "eval" for r in evalset)


def test_stable_ids_are_unchanged_by_splitting() -> None:
    records = _fixture_records(10, per_group=2)
    original_ids = {r["id"] for r in records}
    train, val, evalset, _counts = split_mod.split_records(records, 0.1, 0.1, seed=42)
    result_ids = {r["id"] for r in train} | {r["id"] for r in val} | {r["id"] for r in evalset}
    assert result_ids == original_ids


def test_record_without_knowledge_group_raises() -> None:
    bad_record = _record("x", "kg_000000", ["Page"])
    del bad_record["knowledge_group"]
    with pytest.raises(SystemExit):
        split_mod.split_records([bad_record], 0.1, 0.1, seed=42)


# --- write_jsonl: deterministic ordering + checksums ---------------------


def test_write_jsonl_sorts_by_id(tmp_path: Path) -> None:
    records = [_record("b", "kg_000001", ["P"]), _record("a", "kg_000001", ["P"])]
    out_path = tmp_path / "out.jsonl"
    split_mod.write_jsonl(out_path, records)
    lines = out_path.read_text(encoding="utf-8").splitlines()
    ids = [json.loads(line)["id"] for line in lines]
    assert ids == ["a", "b"]


def test_fixed_seed_and_input_produce_byte_identical_files(tmp_path: Path) -> None:
    records = _fixture_records(12, per_group=2)
    train, val, evalset, _counts = split_mod.split_records(records, 0.2, 0.2, seed=42)

    path1 = tmp_path / "run1.jsonl"
    path2 = tmp_path / "run2.jsonl"
    split_mod.write_jsonl(path1, train)
    split_mod.write_jsonl(path2, train)

    assert path1.read_bytes() == path2.read_bytes()
