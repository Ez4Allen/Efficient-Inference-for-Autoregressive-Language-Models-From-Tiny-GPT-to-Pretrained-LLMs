"""Schemas for the local Terraria guide retrieval corpus."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class WikiPageDescriptor:
    """One page selected by source discovery."""

    title: str
    source_kind: str
    quality_status: str = "unknown"
    quality_flags: list[str] = field(default_factory=list)
    retrieval_role: str = "guide"
    discovery_priority: int = 10_000

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GuideSection:
    """Clean section extracted from a MediaWiki page."""

    section_id: str
    level: int
    title: str
    path: list[str]
    text: str
    character_count: int
    word_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GuideDocument:
    """Clean guide document with provenance and section hierarchy."""

    document_id: str
    page_title: str
    normalized_title: str
    page_id: int | None
    revision_id: int | None
    revision_timestamp: str | None
    source_url: str
    language: str
    source_name: str
    license_name: str
    license_url: str
    retrieved_at: str
    quality_status: str
    quality_flags: list[str]
    categories: list[str]
    sections: list[GuideSection]
    content_sha256: str
    source_kind: str = "unknown"
    retrieval_role: str = "guide"
    discovery_priority: int = 10_000
    parse_status: str = "ok"
    parse_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sections"] = [section.to_dict() for section in self.sections]
        return payload


@dataclass(slots=True)
class GuideChunk:
    """Section-aware retrieval unit."""

    chunk_id: str
    document_id: str
    position: int
    page_title: str
    normalized_title: str
    section_id: str
    section_title: str
    section_path: list[str]
    text: str
    source_url: str
    revision_id: int | None
    language: str
    quality_status: str
    quality_flags: list[str]
    content_sha256: str
    character_count: int
    word_count: int
    source_kind: str = "unknown"
    retrieval_role: str = "guide"
    discovery_priority: int = 10_000
    content_kind: str = "prose"
    table_row_count: int = 0
    table_density: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GuideSearchHit:
    """Public search result returned by :class:`GuideDocumentStore`."""

    chunk_id: str
    document_id: str
    page_title: str
    section_title: str
    section_path: list[str]
    text: str
    source_url: str
    revision_id: int | None
    quality_status: str
    quality_flags: list[str]
    score: float
    rank: int
    matched_terms: list[str] = field(default_factory=list)
    retrieval_role: str = "guide"
    content_kind: str = "prose"
    table_density: float = 0.0

    def citation_label(self) -> str:
        path = " > ".join(self.section_path) if self.section_path else self.section_title
        return f"{self.page_title} — {path}" if path else self.page_title

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["citation_label"] = self.citation_label()
        return payload
