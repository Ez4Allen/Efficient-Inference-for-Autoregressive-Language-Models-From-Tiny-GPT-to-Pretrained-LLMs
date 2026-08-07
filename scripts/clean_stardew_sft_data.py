#!/usr/bin/env python3
"""Clean and audit the legacy Stardew Valley AI-assisted SFT candidate pool.

This utility deliberately does *not* promote candidate records to human-reviewed
training data. It normalizes schema fields, canonicalizes source URLs, assigns a
controlled intent, marks every record as pending review, rejects malformed or
exact duplicate QA pairs, and produces leakage-aware development splits.

The resulting train/validation/eval files are candidate-development partitions,
not the formal benchmark under ``data/stardew/evaluation``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlsplit, urlunsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import read_jsonl, write_json, write_jsonl

ALLOWED_INTENTS = {
    "crop_planning",
    "fish_availability",
    "villager_gifts",
    "recipe_ingredients",
    "bundle_community_center",
    "item_acquisition",
    "guide_progression",
    "other",
}

CATEGORY_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("crop", "seed", "harvest", "seasonal", "farming"), "crop_planning"),
    (("fish", "fishing", "catch", "pond"), "fish_availability"),
    (("villager", "gift", "friendship", "birthday", "marriage"), "villager_gifts"),
    (("recipe", "ingredient", "cooking", "craft", "tailoring"), "recipe_ingredients"),
    (("bundle", "community_center", "community center", "pantry", "fish_tank"), "bundle_community_center"),
    (("acquisition", "purchase", "obtain", "source", "shop", "drop", "location"), "item_acquisition"),
    (("guide", "progress", "strategy", "unlock", "quest", "skill", "mechanic"), "guide_progression"),
)

EN_STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "can", "do", "does", "for", "from",
    "how", "i", "in", "is", "it", "of", "on", "or", "the", "to", "what", "when",
    "where", "which", "with", "you", "your", "stardew", "valley",
}


@dataclass(slots=True)
class Candidate:
    original_index: int
    record: dict[str, Any]
    question: str
    answer: str
    source_pages: tuple[str, ...]
    template_key: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_text(value: str) -> str:
    text = str(value).casefold()
    text = re.sub(r"\d+(?:\.\d+)?", " <num> ", text)
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff<>]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_messages(record: dict[str, Any]) -> tuple[str, str] | None:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return None
    users = [str(item.get("content", "")).strip() for item in messages if isinstance(item, dict) and item.get("role") == "user"]
    assistants = [str(item.get("content", "")).strip() for item in messages if isinstance(item, dict) and item.get("role") == "assistant"]
    if not users or not assistants or not users[-1] or not assistants[-1]:
        return None
    return users[-1], assistants[-1]


def _intent_for(category: str, question: str) -> str:
    haystack = f"{category} {question}".casefold()
    for needles, intent in CATEGORY_RULES:
        if any(needle in haystack for needle in needles):
            return intent
    return "other"


def _canonical_source_url(value: str) -> tuple[str | None, str | None, str | None]:
    raw = str(value).strip()
    if not raw:
        return None, None, "empty_source_url"
    try:
        parts = urlsplit(raw)
    except ValueError:
        return raw, None, "invalid_source_url"
    host = parts.netloc.casefold()
    if not parts.scheme or not host:
        return raw, None, "invalid_source_url"
    path = unquote(parts.path or "/").strip()
    if host in {"www.stardewvalleywiki.com", "stardewvalleywiki.com"}:
        host = "stardewvalleywiki.com"
        title = path.strip("/") or "Main_Page"
        title = re.sub(r"\s+", "_", title)
        canonical = urlunsplit(("https", host, f"/{title}", "", ""))
        page = "stardew:" + re.sub(r"[^a-z0-9]+", "-", title.casefold()).strip("-")
        return canonical, page, None
    canonical = urlunsplit((parts.scheme.casefold(), host, path or "/", "", ""))
    page = f"external:{host}:{hashlib.sha1((path or '/').encode('utf-8')).hexdigest()[:12]}"
    return canonical, page, "non_official_source"


def _question_template(question: str, source_urls: Iterable[str]) -> str:
    text = str(question)
    source_terms: list[str] = []
    for url in source_urls:
        path = unquote(urlsplit(url).path).strip("/").replace("_", " ")
        if path:
            source_terms.append(path)
    for term in sorted(source_terms, key=len, reverse=True):
        text = re.sub(re.escape(term), " <entity> ", text, flags=re.IGNORECASE)
    # Replace multi-token Title Case spans while keeping question words.
    def replace_title(match: re.Match[str]) -> str:
        phrase = match.group(0)
        words = re.findall(r"[A-Za-z]+", phrase)
        if words and all(word.casefold() in EN_STOPWORDS for word in words):
            return phrase
        return " <entity> "

    text = re.sub(r"\b(?:[A-Z][A-Za-z0-9'’-]*)(?:\s+[A-Z][A-Za-z0-9'’-]*)*\b", replace_title, text)
    normalized = _normalize_text(text)
    # The topic prefix avoids collapsing unrelated short templates into one giant group.
    return normalized or "<empty-template>"


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def _connected_groups(candidates: list[Candidate]) -> list[list[Candidate]]:
    dsu = DisjointSet(len(candidates))
    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, item in enumerate(candidates):
        intent = str(item.record["intent"])
        for page in item.source_pages or ("__no_source__",):
            buckets[("source", page)].append(index)
        buckets[("template", f"{intent}:{item.template_key}")].append(index)
    for indexes in buckets.values():
        anchor = indexes[0]
        for index in indexes[1:]:
            dsu.union(anchor, index)
    groups: dict[int, list[Candidate]] = defaultdict(list)
    for index, item in enumerate(candidates):
        groups[dsu.find(index)].append(item)
    return sorted(groups.values(), key=lambda group: (-len(group), min(item.original_index for item in group)))


def _assign_groups(
    groups: list[list[Candidate]],
    *,
    val_fraction: float,
    eval_fraction: float,
    seed: int,
) -> dict[int, str]:
    if not (0 <= val_fraction < 1 and 0 <= eval_fraction < 1):
        raise ValueError("val_fraction and eval_fraction must each be in [0, 1).")
    if val_fraction + eval_fraction >= 1:
        raise ValueError("val_fraction + eval_fraction must be less than 1.")
    total = sum(len(group) for group in groups)
    targets = {
        "validation": round(total * val_fraction),
        "eval": round(total * eval_fraction),
        "train": total - round(total * val_fraction) - round(total * eval_fraction),
    }
    rng = random.Random(seed)
    randomized = list(groups)
    rng.shuffle(randomized)
    randomized.sort(key=len, reverse=True)
    assigned_counts = Counter()
    mapping: dict[int, str] = {}
    for group in randomized:
        # Choose the split with the largest remaining record budget. Stable tie break.
        remaining = {split: targets[split] - assigned_counts[split] for split in ("train", "validation", "eval")}
        split = max(("train", "validation", "eval"), key=lambda name: (remaining[name], -assigned_counts[name], name))
        for item in group:
            mapping[item.original_index] = split
        assigned_counts[split] += len(group)
    return mapping


def clean_records(records: list[dict[str, Any]]) -> tuple[list[Candidate], list[dict[str, Any]], dict[str, Any]]:
    accepted: list[Candidate] = []
    rejected: list[dict[str, Any]] = []
    exact_pairs: set[tuple[str, str]] = set()
    warning_counts: Counter[str] = Counter()
    original_id_counts: Counter[str] = Counter()

    for index, original in enumerate(records):
        record = dict(original)
        original_id = str(record.get("id") or "").strip()
        original_id_counts[original_id] += 1
        parsed = _extract_messages(record)
        reasons: list[str] = []
        if parsed is None:
            reasons.append("missing_valid_user_assistant_messages")
            question = answer = ""
        else:
            question, answer = parsed
        source_urls_raw = record.get("source_urls") or []
        if not isinstance(source_urls_raw, list):
            source_urls_raw = []
            reasons.append("source_urls_not_list")
        canonical_urls: list[str] = []
        pages: list[str] = []
        source_warnings: list[str] = []
        for value in source_urls_raw:
            url, page, warning = _canonical_source_url(str(value))
            if url and url not in canonical_urls:
                canonical_urls.append(url)
            if page and page not in pages:
                pages.append(page)
            if warning:
                source_warnings.append(warning)
                warning_counts[warning] += 1
        if not canonical_urls:
            reasons.append("missing_usable_source_url")
        pair = (_normalize_text(question), _normalize_text(answer))
        if question and answer and pair in exact_pairs:
            reasons.append("exact_duplicate_question_answer")
        elif question and answer:
            exact_pairs.add(pair)
        if reasons:
            rejected.append({**record, "rejection_reasons": sorted(set(reasons)), "original_index": index})
            continue

        original_category = str(record.get("category") or "other").strip() or "other"
        intent = _intent_for(original_category, question)
        if intent not in ALLOWED_INTENTS:
            intent = "other"
        language = str(record.get("language") or ("zh" if re.search(r"[\u4e00-\u9fff]", question) else "en")).strip().casefold()
        if language not in {"en", "zh"}:
            language = "zh" if re.search(r"[\u4e00-\u9fff]", question) else "en"
        cleaned = {
            "id": original_id or f"legacy_{index + 1:04d}",
            "split": "unassigned",
            "game": "stardew_valley",
            "domain": "stardew_valley",
            "category": intent,
            "intent": intent,
            "topic": original_category,
            "language": language,
            "messages": record["messages"],
            "required_facts": list(record.get("required_facts") or []),
            "forbidden_errors": list(record.get("forbidden_errors") or []),
            "source_urls": canonical_urls,
            "source_pages": pages,
            "source_warnings": sorted(set(source_warnings)),
            "verified": False,
            "review_status": "pending",
            "reviewed_by": None,
            "generation_method": str(record.get("generation_method") or "ai_assisted"),
            "dataset_version": "stardew_sft_candidates_v1.1",
            "original_id": original_id or None,
        }
        accepted.append(Candidate(
            original_index=index,
            record=cleaned,
            question=question,
            answer=answer,
            source_pages=tuple(sorted(pages)),
            template_key=_question_template(question, canonical_urls),
        ))

    report = {
        "input_records": len(records),
        "accepted_before_split": len(accepted),
        "rejected_records": len(rejected),
        "duplicate_original_ids": sum(count - 1 for value, count in original_id_counts.items() if value and count > 1),
        "source_warning_counts": dict(sorted(warning_counts.items())),
    }
    return accepted, rejected, report


def run(
    input_path: Path,
    output_dir: Path,
    *,
    val_fraction: float,
    eval_fraction: float,
    seed: int,
) -> dict[str, Any]:
    records = read_jsonl(input_path)
    candidates, rejected, report = clean_records(records)
    groups = _connected_groups(candidates)
    split_map = _assign_groups(groups, val_fraction=val_fraction, eval_fraction=eval_fraction, seed=seed)
    counters = Counter()
    output_rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "eval": []}
    for item in sorted(candidates, key=lambda value: value.original_index):
        split = split_map[item.original_index]
        counters[split] += 1
        row = dict(item.record)
        row["split"] = split
        row["id"] = f"stardew_{split}_{counters[split]:04d}"
        row["template_group"] = hashlib.sha1(f"{row['intent']}:{item.template_key}".encode("utf-8")).hexdigest()[:16]
        output_rows[split].append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in output_rows.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)
    write_jsonl(output_dir / "candidates_clean.jsonl", [row for split in ("train", "validation", "eval") for row in output_rows[split]])
    write_jsonl(output_dir / "rejected.jsonl", rejected)

    pages_by_split = {
        split: {page for row in rows for page in row.get("source_pages") or []}
        for split, rows in output_rows.items()
    }
    templates_by_split = {
        split: {row["template_group"] for row in rows}
        for split, rows in output_rows.items()
    }
    source_overlap = {
        f"{left}_{right}": len(pages_by_split[left] & pages_by_split[right])
        for left, right in (("train", "validation"), ("train", "eval"), ("validation", "eval"))
    }
    template_overlap = {
        f"{left}_{right}": len(templates_by_split[left] & templates_by_split[right])
        for left, right in (("train", "validation"), ("train", "eval"), ("validation", "eval"))
    }
    all_rows = [row for rows in output_rows.values() for row in rows]
    report.update({
        "status": "passed",
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "seed": seed,
        "val_fraction": val_fraction,
        "eval_fraction": eval_fraction,
        "connected_group_count": len(groups),
        "largest_group_size": max((len(group) for group in groups), default=0),
        "split_counts": {split: len(rows) for split, rows in output_rows.items()},
        "language_distribution": dict(Counter(row["language"] for row in all_rows)),
        "intent_distribution": dict(Counter(row["intent"] for row in all_rows)),
        "topic_count": len({row["topic"] for row in all_rows}),
        "unique_source_pages": len({page for row in all_rows for page in row.get("source_pages") or []}),
        "verified_distribution": dict(Counter(str(row["verified"]).casefold() for row in all_rows)),
        "review_status_distribution": dict(Counter(row["review_status"] for row in all_rows)),
        "cross_split_source_overlap": source_overlap,
        "cross_split_template_overlap": template_overlap,
        "output_sha256": {
            name: _sha256(output_dir / name)
            for name in ("train.jsonl", "validation.jsonl", "eval.jsonl", "candidates_clean.jsonl", "rejected.jsonl")
        },
        "notes": [
            "All legacy AI-assisted records are pending human review and verified=false.",
            "These development splits are not the formal evaluation benchmark.",
            "Source-page and normalized-template connected components are kept within a single split.",
        ],
    })
    write_json(output_dir / "audit_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "stardew" / "candidates.jsonl")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "stardew" / "sft")
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--eval-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = run(
        args.input.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        val_fraction=args.val_frac,
        eval_fraction=args.eval_frac,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
