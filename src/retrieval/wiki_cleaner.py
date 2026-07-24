"""Clean rendered MediaWiki HTML into section-aware guide documents."""

from __future__ import annotations

import hashlib
import html as html_module
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from bs4 import BeautifulSoup, NavigableString, Tag

from src.utils.io import read_jsonl, read_yaml, write_json, write_jsonl
from src.utils.paths import portable_path

from .schemas import GuideDocument, GuideSection
from .wiki_importer import utc_now


_SPACE_RE = re.compile(r"[\t\r\f\v ]+")
_BLANK_RE = re.compile(r"\n{3,}")
_REFERENCE_RE = re.compile(r"\[(?:\d+|citation needed|note \d+)\]", re.IGNORECASE)
_EDIT_RE = re.compile(r"\[edit\]", re.IGNORECASE)
_WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)

NOISE_SELECTORS = (
    "script",
    "style",
    "noscript",
    ".mw-editsection",
    ".navbox",
    ".vertical-navbox",
    ".infobox",
    ".metadata",
    ".ambox",
    ".mbox-small",
    ".notice",
    ".noprint",
    ".toc",
    ".mw-empty-elt",
    ".hatnote",
    ".dablink",
    ".gallery",
    "figure",
    ".mw-references-wrap",
    ".reflist",
    ".references",
    ".printfooter",
    ".catlinks",
    ".authority-control",
    ".mw-jump-link",
    ".nomobile",
)


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return normalized or "section"


def _normalize_text(value: str) -> str:
    value = html_module.unescape(str(value)).replace("\xa0", " ")
    value = _EDIT_RE.sub("", value)
    value = _REFERENCE_RE.sub("", value)
    lines: list[str] = []
    for raw_line in value.splitlines():
        line = _SPACE_RE.sub(" ", raw_line).strip()
        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        lines.append(line)
    return _BLANK_RE.sub("\n\n", "\n".join(lines)).strip()


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _table_to_text(table: Tag) -> str:
    rows: list[str] = []
    for row in table.find_all("tr"):
        cells = [
            _normalize_text(cell.get_text(" ", strip=True))
            for cell in row.find_all(["th", "td"], recursive=False)
        ]
        cells = [cell for cell in cells if cell]
        if cells:
            rows.append(" | ".join(cells))
    # Blank lines preserve row boundaries for the paragraph-aware chunker.
    # This prevents large Wiki tables from being sliced in the middle of a row.
    return "\n\n".join(rows)


def _list_to_text(node: Tag, *, ordered: bool) -> str:
    lines: list[str] = []
    for index, item in enumerate(node.find_all("li", recursive=False), start=1):
        text = _normalize_text(item.get_text(" ", strip=True))
        if text:
            prefix = f"{index}." if ordered else "-"
            lines.append(f"{prefix} {text}")
    return "\n".join(lines)


def _description_list_to_text(node: Tag) -> str:
    lines: list[str] = []
    for child in node.children:
        if not isinstance(child, Tag):
            continue
        text = _normalize_text(child.get_text(" ", strip=True))
        if not text:
            continue
        if child.name == "dt":
            lines.append(text)
        elif child.name == "dd":
            lines.append(f"- {text}")
    return "\n".join(lines)


def _extract_blocks(root: Tag) -> list[tuple[str, int | None, str]]:
    """Return ``(kind, heading_level, text)`` blocks in document order."""

    output: list[tuple[str, int | None, str]] = []

    def visit(node: Tag) -> None:
        for child in node.children:
            if isinstance(child, NavigableString):
                continue
            if not isinstance(child, Tag):
                continue
            name = child.name.casefold() if child.name else ""
            if re.fullmatch(r"h[2-6]", name):
                title = _normalize_text(child.get_text(" ", strip=True))
                if title:
                    output.append(("heading", int(name[1]), title))
                continue
            if name == "p":
                text = _normalize_text(child.get_text(" ", strip=True))
                if text:
                    output.append(("text", None, text))
                continue
            if name in {"ul", "ol"}:
                text = _list_to_text(child, ordered=name == "ol")
                if text:
                    output.append(("text", None, text))
                continue
            if name == "dl":
                text = _description_list_to_text(child)
                if text:
                    output.append(("text", None, text))
                continue
            if name == "table":
                text = _table_to_text(child)
                if text:
                    output.append(("text", None, text))
                continue
            if name in {"pre", "blockquote"}:
                text = _normalize_text(child.get_text("\n", strip=True))
                if text:
                    output.append(("text", None, text))
                continue
            if name in {"div", "section", "article"}:
                visit(child)

    visit(root)
    return output


def clean_page_record(
    record: dict[str, Any],
    *,
    excluded_sections: set[str],
    minimum_document_characters: int,
    minimum_section_characters: int,
) -> GuideDocument | None:
    raw_html = record.get("html")
    if not isinstance(raw_html, str) or not raw_html.strip():
        return None

    soup = BeautifulSoup(raw_html, "html.parser")
    root = soup.select_one(".mw-parser-output") or soup
    for selector in NOISE_SELECTORS:
        for node in root.select(selector):
            node.decompose()

    page_title = str(record.get("title") or record.get("requested_title") or "Untitled")
    blocks = _extract_blocks(root)
    paths: dict[int, str] = {}
    sections_raw: list[dict[str, Any]] = [
        {"level": 1, "title": "Overview", "path": ["Overview"], "parts": []}
    ]
    current = sections_raw[0]
    for kind, level, text in blocks:
        if kind == "heading" and level is not None:
            paths[level] = text
            for deeper in list(paths):
                if deeper > level:
                    del paths[deeper]
            section_path = [paths[key] for key in sorted(paths) if key <= level]
            current = {
                "level": level,
                "title": text,
                "path": section_path,
                "parts": [],
            }
            sections_raw.append(current)
        elif kind == "text":
            current["parts"].append(text)

    excluded = {value.casefold().strip() for value in excluded_sections}
    seen_ids: Counter[str] = Counter()
    sections: list[GuideSection] = []
    warnings: list[str] = []
    for section in sections_raw:
        title = str(section["title"]).strip()
        if title.casefold() in excluded:
            continue
        text = _normalize_text("\n\n".join(section["parts"]))
        if not text:
            continue
        if len(text) < minimum_section_characters:
            warnings.append(f"short_section:{title}")
            continue
        base_id = _slug(title)
        seen_ids[base_id] += 1
        section_id = base_id if seen_ids[base_id] == 1 else f"{base_id}-{seen_ids[base_id]}"
        sections.append(
            GuideSection(
                section_id=section_id,
                level=int(section["level"]),
                title=title,
                path=[str(value) for value in section["path"]],
                text=text,
                character_count=len(text),
                word_count=_word_count(text),
            )
        )

    combined_text = "\n\n".join(section.text for section in sections)
    if len(combined_text) < minimum_document_characters:
        return None

    quality_flags = sorted({str(value) for value in record.get("quality_flags", []) if value})
    quality_status = str(record.get("quality_status") or "unknown")
    categories = sorted({str(value) for value in record.get("categories", []) if value})
    category_text = " ".join(categories).casefold()
    if "outdated" in category_text:
        if "outdated_source_code" not in quality_flags:
            quality_flags.append("outdated_source_code")
        warnings.append("quality_category:outdated_source_code")
        if quality_status == "unknown":
            quality_status = "under_revision"
    if page_title.startswith("Legacy:") and "legacy" not in quality_flags:
        quality_flags.append("legacy")
        quality_status = "legacy"

    quality_flags = sorted(set(quality_flags))
    document_id = "wiki:" + hashlib.sha1(page_title.casefold().encode("utf-8")).hexdigest()[:20]
    content_sha256 = hashlib.sha256(combined_text.encode("utf-8")).hexdigest()
    license_info = record.get("license") if isinstance(record.get("license"), dict) else {}
    parse_status = "partial" if warnings else "ok"
    return GuideDocument(
        document_id=document_id,
        page_title=page_title,
        normalized_title=_slug(page_title).replace("-", ""),
        page_id=record.get("page_id"),
        revision_id=record.get("revision_id"),
        revision_timestamp=record.get("revision_timestamp"),
        source_url=str(record.get("source_url") or ""),
        language=str(record.get("language") or "en"),
        source_name=str(record.get("source_name") or "Official Terraria Wiki"),
        license_name=str(license_info.get("name") or "CC BY-NC-SA 4.0"),
        license_url=str(
            license_info.get("url")
            or "https://creativecommons.org/licenses/by-nc-sa/4.0/"
        ),
        retrieved_at=str(record.get("fetched_at") or utc_now()),
        quality_status=quality_status,
        quality_flags=quality_flags,
        categories=categories,
        sections=sections,
        content_sha256=content_sha256,
        source_kind=str(record.get("source_kind") or "unknown"),
        retrieval_role=str(record.get("retrieval_role") or "guide"),
        discovery_priority=int(record.get("discovery_priority", 10_000)),
        parse_status=parse_status,
        parse_warnings=sorted(set(warnings)),
    )


def clean_wiki_pages(
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
    cleaning = manifest.get("cleaning") or {}
    excluded_sections = {str(value) for value in cleaning.get("excluded_sections", [])}
    minimum_document_characters = int(cleaning.get("minimum_document_characters", 200))
    minimum_section_characters = int(cleaning.get("minimum_section_characters", 40))

    raw_records = read_jsonl(input_path)
    documents: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    warning_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    section_count = 0
    for record in raw_records:
        document = clean_page_record(
            record,
            excluded_sections=excluded_sections,
            minimum_document_characters=minimum_document_characters,
            minimum_section_characters=minimum_section_characters,
        )
        if document is None:
            skipped.append(
                {
                    "title": record.get("title") or record.get("requested_title"),
                    "reason": "empty_or_below_minimum_after_cleaning",
                }
            )
            continue
        payload = document.to_dict()
        documents.append(payload)
        section_count += len(document.sections)
        warning_counts.update(document.parse_warnings)
        quality_counts[document.quality_status] += 1

    documents.sort(key=lambda row: str(row["page_title"]).casefold())
    write_jsonl(output_path, documents)
    report = {
        "status": "passed" if not skipped else "partial",
        "input_path": portable_path(input_path),
        "output_path": portable_path(output_path),
        "input_pages": len(raw_records),
        "written_documents": len(documents),
        "skipped_documents": len(skipped),
        "section_count": section_count,
        "parse_status_counts": dict(Counter(row["parse_status"] for row in documents)),
        "quality_status_counts": dict(quality_counts.most_common()),
        "warning_counts": dict(warning_counts.most_common()),
        "skipped": skipped,
        "cleaned_size_bytes": output_path.stat().st_size if output_path.exists() else 0,
        "generated_at": utc_now(),
    }
    write_json(report_path, report)
    return report
