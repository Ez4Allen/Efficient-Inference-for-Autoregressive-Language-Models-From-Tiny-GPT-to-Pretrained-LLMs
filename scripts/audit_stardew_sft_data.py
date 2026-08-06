"""Deterministic, code-generated audit of the Stardew SFT candidate/split data.

Produces a machine-readable JSON report and a human-readable Markdown
summary. Used both as the pre-cleanup baseline (PR1 Step 1) and re-run after
cleanup to show before/after numbers with the same methodology, so the two
reports are directly comparable.

Usage:
    python scripts/audit_stardew_sft_data.py \\
        --split train=data/stardew/sft/train.jsonl \\
        --split validation=data/stardew/sft/validation.jsonl \\
        --split eval=data/stardew/sft/eval.jsonl \\
        --output-json data/stardew/reports/sft_audit_v1.json \\
        --output-md data/stardew/reports/sft_audit_v1.md \\
        --title "Baseline audit (pre-cleanup)"
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


def normalize_text(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


@dataclass
class LoadedRecord:
    split_label: str
    path: str
    line_number: int
    data: dict[str, Any]


@dataclass
class ParseFailure:
    path: str
    line_number: int
    message: str


def load_jsonl(split_label: str, path: Path, failures: list[ParseFailure]) -> list[LoadedRecord]:
    records: list[LoadedRecord] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()

            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as error:
                failures.append(ParseFailure(str(path), line_number, str(error)))
                continue

            if not isinstance(data, dict):
                failures.append(
                    ParseFailure(str(path), line_number, "line is not a JSON object")
                )
                continue

            records.append(LoadedRecord(split_label, str(path), line_number, data))

    return records


def first_content_by_role(messages: Any, role: str) -> str | None:
    if not isinstance(messages, list):
        return None

    for message in messages:
        if isinstance(message, dict) and message.get("role") == role:
            content = message.get("content")
            if isinstance(content, str):
                return content

    return None


def record_source_pages(data: dict[str, Any]) -> list[str]:
    pages = data.get("source_pages")
    if isinstance(pages, list) and pages:
        return [str(page) for page in pages]

    urls = data.get("source_urls")
    if isinstance(urls, list):
        return [str(url) for url in urls]

    return []


@dataclass
class AuditReport:
    title: str
    generated_from: list[dict[str, str]] = field(default_factory=list)
    parse_failures: list[ParseFailure] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def build_report(
    title: str,
    split_paths: dict[str, Path],
    fuzzy_threshold: float,
) -> AuditReport:
    failures: list[ParseFailure] = []
    all_records: list[LoadedRecord] = []

    for label, path in split_paths.items():
        all_records.extend(load_jsonl(label, path, failures))

    report = AuditReport(
        title=title,
        generated_from=[
            {"split": label, "path": str(path)} for label, path in split_paths.items()
        ],
        parse_failures=failures,
    )

    total = len(all_records)

    counts_by_split = Counter(r.split_label for r in all_records)
    counts_by_language = Counter(str(r.data.get("language")) for r in all_records)
    counts_by_category = Counter(str(r.data.get("category")) for r in all_records)
    counts_by_verified = Counter(r.data.get("verified") for r in all_records)
    counts_by_review_status = Counter(
        str(r.data.get("review_status", "<absent>")) for r in all_records
    )

    required_fields = ("id", "messages", "source_urls", "category", "language")
    missing_field_counts: Counter[str] = Counter()

    for r in all_records:
        for field_name in required_fields:
            value = r.data.get(field_name)
            if value in (None, "", []):
                missing_field_counts[field_name] += 1

    # Duplicate record IDs, globally across all provided files.
    id_locations: dict[str, list[str]] = defaultdict(list)
    for r in all_records:
        raw_id = r.data.get("id")
        if isinstance(raw_id, str) and raw_id:
            id_locations[raw_id].append(f"{r.path}:{r.line_number}")

    duplicate_ids = {k: v for k, v in id_locations.items() if len(v) > 1}

    # Exact duplicate question / answer text (raw string, not normalized).
    question_locations: dict[str, list[str]] = defaultdict(list)
    answer_locations: dict[str, list[str]] = defaultdict(list)
    normalized_question_locations: dict[str, list[str]] = defaultdict(list)

    entries: list[tuple[str, str, LoadedRecord]] = []

    for r in all_records:
        question = first_content_by_role(r.data.get("messages"), "user")
        answer = first_content_by_role(r.data.get("messages"), "assistant")
        location = f"{r.path}:{r.line_number}"

        if question:
            question_locations[question].append(location)
            normalized_question_locations[normalize_text(question)].append(location)
            entries.append((normalize_text(question), r.split_label, r))

        if answer:
            answer_locations[answer].append(location)

    exact_duplicate_questions = {k: v for k, v in question_locations.items() if len(v) > 1}
    exact_duplicate_answers = {k: v for k, v in answer_locations.items() if len(v) > 1}
    normalized_duplicate_questions = {
        k: v for k, v in normalized_question_locations.items() if len(v) > 1
    }

    # Source overlap across splits: a canonical source page cited by records
    # living in more than one split bucket.
    pages_by_split: dict[str, set[str]] = defaultdict(set)
    for r in all_records:
        for page in record_source_pages(r.data):
            pages_by_split[r.split_label].add(page.casefold())

    split_labels = sorted(pages_by_split.keys())
    source_overlap: dict[str, list[str]] = {}
    for i, left in enumerate(split_labels):
        for right in split_labels[i + 1:]:
            shared = pages_by_split[left] & pages_by_split[right]
            if shared:
                source_overlap[f"{left}<->{right}"] = sorted(shared)

    source_overlap_count = sum(len(v) for v in source_overlap.values())

    # Near-template overlap across splits: fuzzy-similar questions living in
    # different splits (paraphrase / entity-substitution leakage signal).
    template_overlap_pairs: list[dict[str, str]] = []

    for i in range(len(entries)):
        left_norm, left_split, left_rec = entries[i]
        for j in range(i + 1, len(entries)):
            right_norm, right_split, right_rec = entries[j]

            if left_split == right_split or left_norm == right_norm:
                continue

            if min(len(left_norm), len(right_norm)) < 20:
                continue

            ratio = SequenceMatcher(None, left_norm, right_norm).ratio()

            if ratio >= fuzzy_threshold:
                template_overlap_pairs.append(
                    {
                        "left": f"{left_rec.path}:{left_rec.line_number}",
                        "right": f"{right_rec.path}:{right_rec.line_number}",
                        "similarity": round(ratio, 4),
                    }
                )

    unique_source_pages: set[str] = set()
    for r in all_records:
        for page in record_source_pages(r.data):
            unique_source_pages.add(page.casefold())

    report.stats = {
        "total_records": total,
        "counts_by_split": dict(sorted(counts_by_split.items())),
        "counts_by_language": dict(sorted(counts_by_language.items())),
        "counts_by_verified": {str(k): v for k, v in counts_by_verified.items()},
        "counts_by_review_status": dict(sorted(counts_by_review_status.items())),
        "unique_categories": len(counts_by_category),
        "top_categories": counts_by_category.most_common(20),
        "unique_source_pages": len(unique_source_pages),
        "missing_field_counts": dict(missing_field_counts),
        "duplicate_id_count": len(duplicate_ids),
        "duplicate_ids": {k: v for k, v in list(duplicate_ids.items())[:50]},
        "exact_duplicate_question_groups": len(exact_duplicate_questions),
        "exact_duplicate_question_records": sum(
            len(v) for v in exact_duplicate_questions.values()
        ),
        "exact_duplicate_answer_groups": len(exact_duplicate_answers),
        "exact_duplicate_answer_records": sum(
            len(v) for v in exact_duplicate_answers.values()
        ),
        "normalized_duplicate_question_groups": len(normalized_duplicate_questions),
        "normalized_duplicate_question_records": sum(
            len(v) for v in normalized_duplicate_questions.values()
        ),
        "cross_split_source_overlap_pages": source_overlap_count,
        "cross_split_source_overlap_detail": source_overlap,
        "cross_split_template_overlap_pairs": len(template_overlap_pairs),
        "cross_split_template_overlap_detail": template_overlap_pairs[:100],
        "parse_failure_count": len(failures),
    }

    return report


def write_json(report: AuditReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "title": report.title,
        "generated_from": report.generated_from,
        "parse_failures": [
            {"path": f.path, "line": f.line_number, "message": f.message}
            for f in report.parse_failures
        ],
        "stats": report.stats,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def write_markdown(report: AuditReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    s = report.stats
    lines = [
        f"# {report.title}",
        "",
        "Generated by `scripts/audit_stardew_sft_data.py`. Do not hand-edit; regenerate instead.",
        "",
        "## Source files",
        "",
    ]

    for entry in report.generated_from:
        lines.append(f"- `{entry['split']}`: `{entry['path']}`")

    lines += [
        "",
        "## Headline counts",
        "",
        f"- Total records: **{s['total_records']}**",
        f"- Parse failures: **{s['parse_failure_count']}**",
        f"- Unique categories: **{s['unique_categories']}**",
        f"- Unique canonical source pages: **{s['unique_source_pages']}**",
        f"- Duplicate record IDs: **{s['duplicate_id_count']}**",
        f"- Exact duplicate questions (groups / records): **{s['exact_duplicate_question_groups']} / {s['exact_duplicate_question_records']}**",
        f"- Exact duplicate answers (groups / records): **{s['exact_duplicate_answer_groups']} / {s['exact_duplicate_answer_records']}**",
        f"- Normalized duplicate questions (groups / records): **{s['normalized_duplicate_question_groups']} / {s['normalized_duplicate_question_records']}**",
        f"- Cross-split source-page overlap: **{s['cross_split_source_overlap_pages']}**",
        f"- Cross-split near-template overlap pairs: **{s['cross_split_template_overlap_pairs']}**",
        "",
        "## Counts by split",
        "",
    ]

    for split, count in s["counts_by_split"].items():
        lines.append(f"- {split}: {count}")

    lines += ["", "## Counts by language", ""]
    for lang, count in s["counts_by_language"].items():
        lines.append(f"- {lang}: {count}")

    lines += ["", "## Counts by verified", ""]
    for verified, count in s["counts_by_verified"].items():
        lines.append(f"- {verified}: {count}")

    lines += ["", "## Counts by review_status", ""]
    for status, count in s["counts_by_review_status"].items():
        lines.append(f"- {status}: {count}")

    lines += ["", "## Missing-field counts", ""]
    if s["missing_field_counts"]:
        for f_name, count in s["missing_field_counts"].items():
            lines.append(f"- {f_name}: {count}")
    else:
        lines.append("- none")

    lines += ["", "## Top 20 categories", ""]
    for category, count in s["top_categories"]:
        lines.append(f"- {category}: {count}")

    lines.append("")

    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def parse_split_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Expected label=path, got {value!r}"
        )
    label, _, path_str = value.partition("=")
    return label, Path(path_str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        dest="splits",
        action="append",
        type=parse_split_arg,
        required=True,
        help="label=path, may be repeated (e.g. --split train=data/stardew/sft/train.jsonl)",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--title", type=str, default="Stardew SFT data audit")
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=0.90,
        help="Similarity threshold (0-1) for cross-split template overlap detection.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    split_paths = dict(args.splits)

    report = build_report(
        title=args.title,
        split_paths=split_paths,
        fuzzy_threshold=args.fuzzy_threshold,
    )

    write_json(report, args.output_json)
    write_markdown(report, args.output_md)

    print(f"Total records audited: {report.stats['total_records']}")
    print(f"Parse failures: {report.stats['parse_failure_count']}")
    print(f"JSON report: {args.output_json}")
    print(f"Markdown report: {args.output_md}")

    return 1 if report.parse_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
