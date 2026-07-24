"""Guide-corpus retrieval adapter for progression and mechanics questions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.retrieval.guide_database import (
    DEFAULT_GUIDE_DATABASE_PATH,
    GuideDocumentStore,
)

from .schemas import AssistantRequest, RouteDecision


class DocumentRetriever:
    """Retrieve grounded guide chunks using the local SQLite FTS corpus."""

    def __init__(
        self,
        database_path: str | Path = DEFAULT_GUIDE_DATABASE_PATH,
        *,
        default_limit: int = 6,
        minimum_score: float = 0.14,
        include_low_quality: bool = True,
    ) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.default_limit = max(1, min(int(default_limit), 20))
        self.minimum_score = max(0.0, min(float(minimum_score), 1.0))
        self.include_low_quality = bool(include_low_quality)
        self.store: GuideDocumentStore | None = None
        if self.database_path.exists():
            self.store = GuideDocumentStore(self.database_path)

    @property
    def available(self) -> bool:
        return self.store is not None

    def close(self) -> None:
        if self.store is not None:
            self.store.close()
            self.store = None

    def retrieve(
        self,
        request: AssistantRequest,
        route: RouteDecision,
    ) -> dict[str, Any]:
        if self.store is None:
            return {
                "status": "not_found",
                "intent": "guide",
                "query": request.question,
                "facts": None,
                "candidates": [],
                "warnings": [
                    "The local Terraria guide corpus is not built. Run "
                    "`python scripts/build_terraria_guides.py` before asking "
                    "progression or mechanics questions."
                ],
                "provenance": [],
            }

        limit = int(route.parameters.get("guide_limit", self.default_limit))
        minimum_score = float(
            route.parameters.get("guide_minimum_score", self.minimum_score)
        )
        hits = self.store.search(
            request.question,
            limit=limit,
            minimum_score=minimum_score,
            include_low_quality=self.include_low_quality,
        )
        if not hits:
            return {
                "status": "not_found",
                "intent": "guide",
                "query": request.question,
                "facts": None,
                "candidates": [],
                "warnings": [
                    "No sufficiently relevant local guide evidence was found, "
                    "so the assistant will not invent progression or mechanics advice."
                ],
                "provenance": [],
            }

        warnings: list[str] = []
        low_quality_hits = [
            hit
            for hit in hits
            if hit.get("quality_status") in {"under_revision", "subject_to_revision", "legacy"}
        ]
        if low_quality_hits:
            warnings.append(
                "Some retrieved Wiki guide sections are marked as under revision, "
                "subject to revision, or legacy."
            )

        top_score = float(hits[0].get("score", 0.0))
        facts = {
            "query": request.question,
            "hit_count": len(hits),
            "top_score": top_score,
            "retrieval_method": "sqlite_fts5_bilingual_query_expansion",
            "hits": hits,
        }
        provenance = [
            {
                "entity_type": "guide_chunk",
                "source_catalog_id": hit["chunk_id"],
                "document_id": hit["document_id"],
                "page_title": hit["page_title"],
                "section_title": hit["section_title"],
                "source_url": hit["source_url"],
                "revision_id": hit.get("revision_id"),
                "quality_status": hit.get("quality_status"),
                "score": hit.get("score"),
            }
            for hit in hits
        ]
        return {
            "status": "found",
            "intent": "guide",
            "query": request.question,
            "facts": facts,
            "candidates": [],
            "warnings": warnings,
            "provenance": provenance,
        }
