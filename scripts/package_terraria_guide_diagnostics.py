#!/usr/bin/env python3
"""Package lightweight guide-pipeline diagnostics for external review."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.guide_database import GuideDocumentStore
from src.retrieval.pipeline import DEFAULT_GUIDES_ROOT
from src.utils.io import read_jsonl, write_json, write_jsonl


def _sample_evenly(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(records) <= limit:
        return records
    if limit <= 1:
        return [records[0]]
    indices = {round(index * (len(records) - 1) / (limit - 1)) for index in range(limit)}
    return [records[index] for index in sorted(indices)]


def _trim_document(record: dict[str, Any]) -> dict[str, Any]:
    payload = dict(record)
    payload["sections"] = [
        {
            **section,
            "text": str(section.get("text") or "")[:1200],
            "text_truncated_for_diagnostics": len(str(section.get("text") or "")) > 1200,
        }
        for section in (record.get("sections") or [])[:8]
    ]
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a small ZIP containing reports and samples for guide-cleaning review."
    )
    parser.add_argument("--guides-root", type=Path, default=DEFAULT_GUIDES_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/content/terraria_guide_diagnostics.zip"),
    )
    parser.add_argument("--document-samples", type=int, default=20)
    parser.add_argument("--chunk-samples", type=int, default=50)
    args = parser.parse_args()

    root = args.guides_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    documents_path = root / "cleaned" / "documents.jsonl"
    chunks_path = root / "chunks" / "chunks.jsonl"
    if not documents_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(
            "Build the guide corpus before packaging diagnostics. Missing cleaned documents or chunks."
        )

    documents = read_jsonl(documents_path)
    chunks = read_jsonl(chunks_path)
    samples_root = root / "reports" / ".diagnostic_samples"
    samples_root.mkdir(parents=True, exist_ok=True)

    sampled_documents = [
        _trim_document(record)
        for record in _sample_evenly(documents, max(1, args.document_samples))
    ]
    suspicious = [
        _trim_document(record)
        for record in documents
        if record.get("parse_status") != "ok"
        or record.get("parse_warnings")
        or record.get("quality_status") in {"under_revision", "subject_to_revision", "legacy"}
    ][:100]
    sampled_chunks = _sample_evenly(chunks, max(1, args.chunk_samples))
    suspicious_chunks = [
        record
        for record in chunks
        if record.get("content_kind") == "table"
        or float(record.get("table_density") or 0.0) >= 0.65
        or record.get("quality_status") in {"under_revision", "subject_to_revision", "legacy"}
        or "outdated_source_code" in (record.get("quality_flags") or [])
    ][:100]

    document_sample_path = samples_root / "sample_documents.jsonl"
    suspicious_path = samples_root / "suspicious_documents.jsonl"
    chunk_sample_path = samples_root / "sample_chunks.jsonl"
    suspicious_chunk_path = samples_root / "suspicious_chunks.jsonl"
    retrieval_probe_path = samples_root / "retrieval_probes.json"
    write_jsonl(document_sample_path, sampled_documents)
    write_jsonl(suspicious_path, suspicious)
    write_jsonl(chunk_sample_path, sampled_chunks)
    write_jsonl(suspicious_chunk_path, suspicious_chunks)

    probes = {}
    database_path = root / "terraria_guides.sqlite3"
    if database_path.exists():
        queries = [
            "What should I do on the first night?",
            "What should I do after entering Hardmode?",
            "What is the recommended boss progression?",
            "How do I control biome spread?",
            "进入困难模式后该做什么？",
        ]
        with GuideDocumentStore(database_path) as store:
            probes = {query: store.search(query, limit=5) for query in queries}
    write_json(retrieval_probe_path, probes)

    readme = samples_root / "README.txt"
    readme.write_text(
        "This archive contains lightweight reports and samples only.\n"
        "It excludes full raw HTML, the full guide corpus, and SQLite databases.\n"
        "Upload this ZIP for cleaning and retrieval review.\n",
        encoding="utf-8",
    )

    candidates = [
        root / "config" / "sources.yaml",
        root / "ATTRIBUTION.md",
        *sorted((root / "reports").glob("*.json")),
        document_sample_path,
        suspicious_path,
        chunk_sample_path,
        suspicious_chunk_path,
        retrieval_probe_path,
        readme,
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in candidates:
            if not path.exists():
                continue
            archive.write(path, arcname=path.relative_to(root).as_posix())

    print(output)
    print(f"Size: {output.stat().st_size / 1024:.2f} KB")


if __name__ == "__main__":
    main()
