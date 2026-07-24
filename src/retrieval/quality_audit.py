"""Static quality audit for cleaned Terraria guide documents and chunks."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

from src.utils.io import read_jsonl, write_json
from src.utils.paths import portable_path

from .wiki_importer import utc_now


MARKUP_PATTERNS = {
    "html_tag": re.compile(r"<\/?[A-Za-z][^>]*>"),
    "wikilink": re.compile(r"\[\[[^\]]+\]\]"),
    "template": re.compile(r"\{\{[^}]+\}\}"),
    "navigation_text": re.compile(
        r"jump to navigation|retrieved from|categories?:\s*$",
        flags=re.IGNORECASE | re.MULTILINE,
    ),
}


def audit_guide_corpus(
    *,
    documents_path: str | Path,
    chunks_path: str | Path,
    report_path: str | Path,
    sample_limit: int = 50,
) -> dict[str, Any]:
    documents_path = Path(documents_path).expanduser().resolve()
    chunks_path = Path(chunks_path).expanduser().resolve()
    report_path = Path(report_path).expanduser().resolve()
    documents = read_jsonl(documents_path)
    chunks = read_jsonl(chunks_path)

    document_ids = {row["document_id"] for row in documents}
    chunk_ids: set[str] = set()
    content_hashes: set[str] = set()
    warning_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    suspicious: list[dict[str, Any]] = []
    orphan_count = 0
    duplicate_id_count = 0
    duplicate_content_count = 0

    for chunk in chunks:
        reasons: list[str] = []
        chunk_id = str(chunk.get("chunk_id") or "")
        if chunk_id in chunk_ids:
            duplicate_id_count += 1
            reasons.append("duplicate_chunk_id")
        chunk_ids.add(chunk_id)
        content_hash = str(chunk.get("content_sha256") or "")
        if content_hash in content_hashes:
            duplicate_content_count += 1
            reasons.append("duplicate_chunk_content")
        content_hashes.add(content_hash)

        if chunk.get("document_id") not in document_ids:
            orphan_count += 1
            reasons.append("orphan_document_reference")

        text = str(chunk.get("text") or "")
        if len(text) < 40:
            reasons.append("very_short_chunk")
        if len(text) > 2600:
            reasons.append("oversized_chunk")
        for name, pattern in MARKUP_PATTERNS.items():
            if pattern.search(text):
                reasons.append(f"markup_remnant:{name}")

        content_kind = str(chunk.get("content_kind") or "prose")
        table_density = float(chunk.get("table_density") or 0.0)
        if content_kind == "table" or table_density >= 0.65:
            reasons.append("content:table_heavy")
        if text and text[0].islower() and chunk.get("position", 0) > 1:
            reasons.append("possible_mid_sentence_start")

        quality_status = str(chunk.get("quality_status") or "unknown")
        quality_counts[quality_status] += 1
        if quality_status in {"under_revision", "subject_to_revision", "legacy"}:
            reasons.append(f"quality:{quality_status}")
        for flag in chunk.get("quality_flags") or []:
            if flag == "outdated_source_code":
                reasons.append("quality:outdated_source_code")

        if reasons:
            warning_counts.update(reasons)
            if len(suspicious) < max(1, int(sample_limit)):
                suspicious.append(
                    {
                        "chunk_id": chunk_id,
                        "page_title": chunk.get("page_title"),
                        "section_title": chunk.get("section_title"),
                        "reasons": reasons,
                        "text_preview": text[:500],
                        "source_url": chunk.get("source_url"),
                    }
                )

    critical_counts = {
        "orphan_document_references": orphan_count,
        "duplicate_chunk_ids": duplicate_id_count,
        "duplicate_chunk_content": duplicate_content_count,
    }
    status = "passed" if all(value == 0 for value in critical_counts.values()) else "failed"
    report = {
        "status": status,
        "documents_path": portable_path(documents_path),
        "chunks_path": portable_path(chunks_path),
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "documents_with_chunks": len({row.get("document_id") for row in chunks}),
        "quality_status_counts": dict(quality_counts.most_common()),
        "warning_counts": dict(warning_counts.most_common()),
        "critical_counts": critical_counts,
        "suspicious_samples": suspicious,
        "generated_at": utc_now(),
    }
    write_json(report_path, report)
    if status != "passed":
        raise AssertionError(f"Guide corpus quality audit failed: {critical_counts}")
    return report
