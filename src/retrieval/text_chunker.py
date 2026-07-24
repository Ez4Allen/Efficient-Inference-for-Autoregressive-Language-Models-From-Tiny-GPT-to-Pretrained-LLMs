"""Deterministic paragraph- and table-aware chunking for Terraria guides."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Any

from src.utils.io import read_jsonl, read_yaml, write_json, write_jsonl
from src.utils.paths import portable_path

from .wiki_importer import utc_now

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")
_WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
_TABLE_SEPARATOR_RE = re.compile(r"\s+\|\s+")


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _table_metrics(text: str) -> tuple[int, float]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0, 0.0
    table_lines = [line for line in lines if line.count(" | ") >= 1]
    table_characters = sum(len(line) for line in table_lines)
    total_characters = sum(len(line) for line in lines)
    return len(table_lines), round(table_characters / max(1, total_characters), 6)


def _content_kind(text: str) -> tuple[str, int, float]:
    row_count, density = _table_metrics(text)
    if row_count >= 2 and density >= 0.45:
        return "table", row_count, density
    if row_count:
        return "mixed", row_count, density
    return "prose", 0, 0.0


def _pack_units(units: list[str], maximum_characters: int, separator: str) -> list[str]:
    output: list[str] = []
    current: list[str] = []
    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        candidate = separator.join([*current, unit])
        if current and len(candidate) > maximum_characters:
            output.append(separator.join(current).strip())
            current = []
        if len(unit) <= maximum_characters:
            current.append(unit)
            continue
        # Final fallback for an indivisible unit. It is used only after
        # sentence, line, and table-cell boundaries have been exhausted.
        output.extend(
            unit[index : index + maximum_characters].strip()
            for index in range(0, len(unit), maximum_characters)
            if unit[index : index + maximum_characters].strip()
        )
    if current:
        output.append(separator.join(current).strip())
    return output


def _split_table_row(text: str, maximum_characters: int) -> list[str]:
    cells = [value.strip() for value in _TABLE_SEPARATOR_RE.split(text) if value.strip()]
    if len(cells) <= 1:
        return _pack_units([text], maximum_characters, " ")
    # Repeat the row label when one logical row must span multiple chunks.
    label = cells[0]
    pieces: list[str] = []
    current = label
    for cell in cells[1:]:
        candidate = f"{current} | {cell}"
        if len(candidate) <= maximum_characters:
            current = candidate
            continue
        if current != label:
            pieces.append(current)
        current = f"{label} | {cell}"
        if len(current) > maximum_characters:
            pieces.extend(_pack_units([current], maximum_characters, " "))
            current = label
    if current != label:
        pieces.append(current)
    return pieces or [text[:maximum_characters].strip()]


def _split_large_block(text: str, maximum_characters: int) -> list[str]:
    if len(text) <= maximum_characters:
        return [text]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        expanded: list[str] = []
        for line in lines:
            expanded.extend(_split_large_block(line, maximum_characters))
        return _pack_units(expanded, maximum_characters, "\n")

    if text.count(" | ") >= 2:
        return _split_table_row(text, maximum_characters)

    sentences = [value.strip() for value in _SENTENCE_SPLIT_RE.split(text) if value.strip()]
    if len(sentences) > 1:
        return _pack_units(sentences, maximum_characters, " ")

    # Prefer word boundaries to raw character slicing.
    words = text.split()
    if len(words) > 1:
        return _pack_units(words, maximum_characters, " ")

    return [
        text[index : index + maximum_characters].strip()
        for index in range(0, len(text), maximum_characters)
        if text[index : index + maximum_characters].strip()
    ]


def _paragraphs(text: str, maximum_characters: int) -> list[str]:
    raw = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    output: list[str] = []
    for block in raw:
        output.extend(_split_large_block(block, maximum_characters))
    return output


def _overlap_tail(parts: list[str], overlap_characters: int) -> list[str]:
    if overlap_characters <= 0:
        return []
    selected: list[str] = []
    total = 0
    for part in reversed(parts):
        selected.append(part)
        total += len(part)
        if total >= overlap_characters:
            break
    selected.reverse()
    return selected


def _section_chunks(
    text: str,
    *,
    maximum_characters: int,
    overlap_characters: int,
) -> list[str]:
    parts = _paragraphs(text, maximum_characters)
    chunks: list[str] = []
    current_parts: list[str] = []
    for part in parts:
        candidate_parts = [*current_parts, part]
        candidate = "\n\n".join(candidate_parts)
        if current_parts and len(candidate) > maximum_characters:
            chunks.append("\n\n".join(current_parts).strip())
            current_parts = _overlap_tail(current_parts, overlap_characters)
            while current_parts and len("\n\n".join([*current_parts, part])) > maximum_characters:
                current_parts.pop(0)
        current_parts.append(part)
    if current_parts:
        chunks.append("\n\n".join(current_parts).strip())
    return [chunk for chunk in chunks if chunk]


def chunk_guide_documents(
    *,
    input_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    report_path = Path(report_path).expanduser().resolve()
    manifest = read_yaml(manifest_path)
    chunking = manifest.get("chunking") or {}
    maximum_characters = int(chunking.get("maximum_characters", 1800))
    overlap_characters = int(chunking.get("overlap_characters", 180))
    minimum_characters = int(chunking.get("minimum_characters", 120))
    table_maximum_characters = int(
        chunking.get("table_maximum_characters", min(maximum_characters, 1200))
    )
    if maximum_characters < 200:
        raise ValueError("maximum_characters must be at least 200.")
    if table_maximum_characters < 200:
        raise ValueError("table_maximum_characters must be at least 200.")
    if not 0 <= overlap_characters < maximum_characters:
        raise ValueError("overlap_characters must be >= 0 and < maximum_characters.")

    documents = read_jsonl(input_path)
    chunks: list[dict[str, Any]] = []
    seen_hashes: dict[str, str] = {}
    duplicate_count = 0
    short_count = 0
    quality_counts: Counter[str] = Counter()
    content_kind_counts: Counter[str] = Counter()

    for document in documents:
        position = 0
        for section in document.get("sections", []) or []:
            section_text = str(section.get("text") or "")
            section_kind, _, section_table_density = _content_kind(section_text)
            effective_maximum = (
                table_maximum_characters if section_table_density >= 0.45 else maximum_characters
            )
            effective_overlap = 0 if section_kind == "table" else overlap_characters
            raw_chunks = _section_chunks(
                section_text,
                maximum_characters=effective_maximum,
                overlap_characters=effective_overlap,
            )
            if len(raw_chunks) > 1 and len(raw_chunks[-1]) < minimum_characters:
                merged = f"{raw_chunks[-2]}\n\n{raw_chunks[-1]}".strip()
                if len(merged) <= int(effective_maximum * 1.2):
                    raw_chunks[-2:] = [merged]

            for local_index, text in enumerate(raw_chunks, start=1):
                if len(text) < minimum_characters:
                    short_count += 1
                    if len(text) < max(40, minimum_characters // 2):
                        continue
                content_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if content_sha256 in seen_hashes:
                    duplicate_count += 1
                    continue
                position += 1
                chunk_id = (
                    f"{document['document_id']}::{section['section_id']}::{local_index:03d}"
                )
                seen_hashes[content_sha256] = chunk_id
                quality_status = str(document.get("quality_status") or "unknown")
                quality_counts[quality_status] += 1
                content_kind, table_row_count, table_density = _content_kind(text)
                content_kind_counts[content_kind] += 1
                chunks.append(
                    {
                        "schema_version": 2,
                        "chunk_id": chunk_id,
                        "document_id": document["document_id"],
                        "position": position,
                        "page_title": document["page_title"],
                        "normalized_title": document["normalized_title"],
                        "section_id": section["section_id"],
                        "section_title": section["title"],
                        "section_path": section.get("path") or [section["title"]],
                        "text": text,
                        "source_url": document["source_url"],
                        "revision_id": document.get("revision_id"),
                        "language": document.get("language", "en"),
                        "quality_status": quality_status,
                        "quality_flags": document.get("quality_flags") or [],
                        "source_kind": document.get("source_kind", "unknown"),
                        "retrieval_role": document.get("retrieval_role", "guide"),
                        "discovery_priority": int(document.get("discovery_priority", 10_000)),
                        "content_kind": content_kind,
                        "table_row_count": table_row_count,
                        "table_density": table_density,
                        "content_sha256": content_sha256,
                        "character_count": len(text),
                        "word_count": _word_count(text),
                        "license_name": document.get("license_name"),
                        "license_url": document.get("license_url"),
                    }
                )

    chunks.sort(
        key=lambda row: (
            int(row.get("discovery_priority", 10_000)),
            str(row["page_title"]).casefold(),
            int(row["position"]),
        )
    )
    write_jsonl(output_path, chunks)
    report = {
        "status": "passed",
        "input_path": portable_path(input_path),
        "output_path": portable_path(output_path),
        "input_documents": len(documents),
        "written_chunks": len(chunks),
        "duplicate_chunks_skipped": duplicate_count,
        "short_chunks_observed": short_count,
        "quality_status_counts": dict(quality_counts.most_common()),
        "content_kind_counts": dict(content_kind_counts.most_common()),
        "minimum_characters": minimum_characters,
        "maximum_characters": maximum_characters,
        "table_maximum_characters": table_maximum_characters,
        "overlap_characters": overlap_characters,
        "chunk_size_bytes": output_path.stat().st_size if output_path.exists() else 0,
        "generated_at": utc_now(),
    }
    write_json(report_path, report)
    return report
