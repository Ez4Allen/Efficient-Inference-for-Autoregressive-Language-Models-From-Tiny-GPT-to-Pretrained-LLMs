"""Split the cleaned Stardew SFT candidate pool into train/validation/eval.

Consumes the normalized pool produced by ``scripts/clean_stardew_sft_data.py``
(``data/stardew/sft/candidates.normalized.jsonl``). Records with
``review_status == "rejected"`` are excluded (see
``data/stardew/sft/rejected.jsonl`` for why). Remaining records keep
``verified: false`` / ``review_status: pending`` unless they carry auditable
review evidence -- this splitter does not certify data as human-verified,
it only partitions it without leaking a knowledge family across splits.

Splitting is done by ``knowledge_group`` (assigned by the cleaning script
from canonical source-page overlap plus entity-substitution template
similarity), not by individual record or raw URL, so every record derived
from the same wiki page or the same underlying question template lands in
one split.

Usage:
    python scripts/split_stardew_sft_data.py \\
        --input data/stardew/sft/candidates.normalized.jsonl \\
        --outdir data/stardew/sft \\
        --manifest data/stardew/reports/sft_split_manifest.json \\
        --val-frac 0.1 --eval-frac 0.1 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.games.stardew.sft_text_utils import mask_template, normalize_text  # noqa: E402
from difflib import SequenceMatcher  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"Invalid JSON at {path}:{line_number}: {error}")
            if not isinstance(data, dict):
                raise SystemExit(f"{path}:{line_number} is not a JSON object")
            records.append(data)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda r: r["id"])
    with path.open("w", encoding="utf-8") as handle:
        for record in ordered:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_fractions(val_frac: float, eval_frac: float) -> None:
    if not (0.0 <= val_frac < 1.0):
        raise SystemExit(f"--val-frac must satisfy 0 <= val_frac < 1, got {val_frac}")
    if not (0.0 <= eval_frac < 1.0):
        raise SystemExit(f"--eval-frac must satisfy 0 <= eval_frac < 1, got {eval_frac}")
    if val_frac + eval_frac >= 1.0:
        raise SystemExit(
            "val_frac + eval_frac must be < 1 so train is non-empty, got "
            f"val_frac={val_frac}, eval_frac={eval_frac}"
        )


def compute_group_split_counts(
    n_groups: int,
    val_frac: float,
    eval_frac: float,
) -> tuple[int, int, int]:
    """Return ``(n_train, n_val, n_eval)`` group counts.

    Small group counts (1, 2, 3) are handled explicitly rather than through
    the general rounding formula, because rounding a fraction of a tiny
    denominator either drops the held-out split entirely or -- with the
    previous implementation's ``max(1, int(...))`` approach -- can demand
    more groups than exist and drive the train count negative.
    """

    if n_groups <= 0:
        raise SystemExit("No knowledge groups available to split; nothing to do.")

    wants_val = val_frac > 0
    wants_eval = eval_frac > 0

    if n_groups == 1:
        return 1, 0, 0

    if n_groups == 2:
        if wants_val and wants_eval:
            return (1, 0, 1) if eval_frac >= val_frac else (1, 1, 0)
        if wants_eval:
            return 1, 0, 1
        if wants_val:
            return 1, 1, 0
        return 2, 0, 0

    if n_groups == 3:
        if wants_val and wants_eval:
            return 1, 1, 1
        if wants_eval:
            return 2, 0, 1
        if wants_val:
            return 2, 1, 0
        return 3, 0, 0

    n_eval = round(n_groups * eval_frac)
    n_val = round(n_groups * val_frac)

    if wants_eval and n_eval == 0:
        n_eval = 1
    if wants_val and n_val == 0:
        n_val = 1

    n_eval = min(n_eval, n_groups - 1)
    n_val = min(n_val, max(n_groups - n_eval - 1, 0))
    n_train = n_groups - n_eval - n_val

    if n_train < 1:
        raise SystemExit(
            "Cannot allocate at least one knowledge group to train with "
            f"val_frac={val_frac}, eval_frac={eval_frac}, n_groups={n_groups}."
        )

    return n_train, n_val, n_eval


def split_records(
    records: list[dict[str, Any]],
    val_frac: float,
    eval_frac: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        group_id = record.get("knowledge_group")
        if not group_id:
            raise SystemExit(
                f"Record {record.get('id')!r} has no knowledge_group; "
                "run scripts/clean_stardew_sft_data.py first."
            )
        groups[group_id].append(record)

    group_ids = sorted(groups.keys())
    n_train, n_val, n_eval = compute_group_split_counts(len(group_ids), val_frac, eval_frac)

    rng = random.Random(seed)
    shuffled = list(group_ids)
    rng.shuffle(shuffled)

    eval_groups = set(shuffled[:n_eval])
    val_groups = set(shuffled[n_eval:n_eval + n_val])
    train_groups = set(shuffled[n_eval + n_val:])

    train, validation, evalset = [], [], []
    for group_id in group_ids:
        for record in groups[group_id]:
            record = dict(record)
            if group_id in eval_groups:
                record["split"] = "eval"
                evalset.append(record)
            elif group_id in val_groups:
                record["split"] = "validation"
                validation.append(record)
            else:
                record["split"] = "train"
                train.append(record)

    group_counts = {
        "total": len(group_ids),
        "train": len(train_groups),
        "validation": len(val_groups),
        "eval": len(eval_groups),
    }

    return train, validation, evalset, group_counts


def compute_overlap_diagnostics(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    evalset: list[dict[str, Any]],
    fuzzy_threshold: float,
) -> dict[str, Any]:
    splits = {"train": train, "validation": validation, "eval": evalset}

    pages_by_split: dict[str, set[str]] = {}
    for label, recs in splits.items():
        pages = set()
        for record in recs:
            for page in record.get("source_pages", []):
                pages.add(str(page).casefold())
        pages_by_split[label] = pages

    source_overlap = 0
    labels = list(splits.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            source_overlap += len(pages_by_split[labels[i]] & pages_by_split[labels[j]])

    # Two distinct cross-split metrics, matching the two-tier detection used
    # by the cleaning pipeline (see clean_stardew_sft_data.py):
    #   - near-duplicate overlap (unmasked question text): the critical
    #     safety number, expected to be ~0 -- this is real leakage, the same
    #     question effectively answered in two different splits.
    #   - template-shape overlap (entity-masked question text): informative
    #     only. This dataset relies heavily on a handful of generic
    #     acquisition-style phrasings across hundreds of distinct entities,
    #     so this number is expected to stay large; forcing it to zero would
    #     require merging most of item_acquisition into one knowledge group
    #     (see the cleaning script's docstring for why that was rejected).
    entries: list[tuple[str, str, str]] = []
    for label, recs in splits.items():
        for record in recs:
            for message in record.get("messages", []):
                if isinstance(message, dict) and message.get("role") == "user":
                    content = message.get("content")
                    if isinstance(content, str):
                        entities = [
                            p.replace("_", " ") for p in record.get("source_pages", [])
                        ]
                        entries.append(
                            (label, normalize_text(content), mask_template(content, entities))
                        )

    near_duplicate_overlap = 0
    template_overlap = 0

    for i in range(len(entries)):
        left_label, left_unmasked, left_masked = entries[i]
        for j in range(i + 1, len(entries)):
            right_label, right_unmasked, right_masked = entries[j]
            if left_label == right_label:
                continue

            if len(left_unmasked) >= 20 and len(right_unmasked) >= 20:
                if (
                    SequenceMatcher(None, left_unmasked, right_unmasked).ratio()
                    >= fuzzy_threshold
                ):
                    near_duplicate_overlap += 1

            if len(left_masked) >= 20 and len(right_masked) >= 20:
                if SequenceMatcher(None, left_masked, right_masked).ratio() >= fuzzy_threshold:
                    template_overlap += 1

    return {
        "cross_split_source_overlap_count": source_overlap,
        "cross_split_near_duplicate_overlap_count": near_duplicate_overlap,
        "cross_split_template_overlap_count": template_overlap,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/stardew/sft/candidates.normalized.jsonl")
    )
    parser.add_argument("--outdir", type=Path, default=Path("data/stardew/sft"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/stardew/reports/sft_split_manifest.json")
    )
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--eval-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=0.90,
        help="Similarity threshold for the post-split overlap diagnostic (informational only).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_fractions(args.val_frac, args.eval_frac)

    all_records = load_jsonl(args.input)
    kept = [r for r in all_records if r.get("review_status") != "rejected"]

    train, validation, evalset, group_counts = split_records(
        kept, args.val_frac, args.eval_frac, args.seed
    )

    train_path = args.outdir / "train.jsonl"
    val_path = args.outdir / "validation.jsonl"
    eval_path = args.outdir / "eval.jsonl"

    write_jsonl(train_path, train)
    write_jsonl(val_path, validation)
    write_jsonl(eval_path, evalset)

    overlap = compute_overlap_diagnostics(train, validation, evalset, args.fuzzy_threshold)

    manifest = {
        "seed": args.seed,
        "val_frac": args.val_frac,
        "eval_frac": args.eval_frac,
        "input_path": str(args.input),
        "input_record_count": len(all_records),
        "excluded_rejected_count": len(all_records) - len(kept),
        "knowledge_group_counts": group_counts,
        "record_counts": {
            "train": len(train),
            "validation": len(validation),
            "eval": len(evalset),
        },
        "file_checksums_sha256": {
            "train": sha256_of_file(train_path),
            "validation": sha256_of_file(val_path),
            "eval": sha256_of_file(eval_path),
        },
        **overlap,
    }

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")

    print(f"Knowledge groups: {group_counts}")
    print(f"Records: train {len(train)}, validation {len(validation)}, eval {len(evalset)}")
    print(f"Cross-split source overlap: {overlap['cross_split_source_overlap_count']}")
    print(
        "Cross-split near-duplicate overlap (critical): "
        f"{overlap['cross_split_near_duplicate_overlap_count']}"
    )
    print(
        "Cross-split template-shape overlap (informational): "
        f"{overlap['cross_split_template_overlap_count']}"
    )
    print(f"Manifest written to {args.manifest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
