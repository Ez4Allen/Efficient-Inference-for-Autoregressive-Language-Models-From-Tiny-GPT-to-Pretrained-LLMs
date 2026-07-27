"""End-to-end deterministic Stardew Valley assistant plug-in."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.gameguide.schemas import GameEvidence, GameGuideResult
from src.utils.paths import STARDEW_CATALOG_ROOT

from .database_builder import DEFAULT_DATABASE_PATH, build_stardew_database
from .fact_service import StardewFactService
from .guide_pipeline import (
    DEFAULT_DATABASE_PATH as DEFAULT_GUIDE_DATABASE_PATH,
    build_stardew_guides,
    build_stardew_seed_guides,
)
from .guide_store import StardewGuideStore
from .intent_router import StardewIntentRouter
from .renderer import StardewRenderer


class StardewAssistant:
    game_id = "stardew_valley"
    display_name = "Stardew Valley"

    def __init__(
        self,
        *,
        database_path: str | Path = DEFAULT_DATABASE_PATH,
        guide_database_path: str | Path = DEFAULT_GUIDE_DATABASE_PATH,
        auto_build: bool = True,
    ) -> None:
        database_path = Path(database_path)
        if auto_build and not database_path.exists():
            build_stardew_database(database_path=database_path)
        self.service = StardewFactService(database_path)
        self.router = StardewIntentRouter(self.service.store)
        self.renderer = StardewRenderer()
        guide_path = Path(guide_database_path)
        if auto_build and not guide_path.exists():
            raw_snapshot = guide_path.parent / "raw" / "pages.jsonl"
            if raw_snapshot.exists():
                build_stardew_guides(
                    guides_root=guide_path.parent,
                    offline=True,
                    verbose=False,
                )
            else:
                seed_path = guide_path.parent / "seed" / "pages.jsonl"
                if seed_path.exists():
                    build_stardew_seed_guides(
                        guides_root=guide_path.parent,
                        seed_path=seed_path,
                        verbose=False,
                    )
        self.guide_store = StardewGuideStore(guide_path) if guide_path.exists() else None
        self._closed = False

    def __enter__(self) -> "StardewAssistant":
        if self._closed:
            raise RuntimeError("StardewAssistant is closed.")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            if self.guide_store is not None:
                self.guide_store.close()
            self.service.close()
            self._closed = True

    @staticmethod
    def _evidence_from_provenance(provenance: list[dict[str, Any]]) -> list[GameEvidence]:
        evidence = []
        seen = set()
        for item in provenance:
            key = str(item.get("source_catalog_id") or item.get("source_url"))
            if not key or key in seen:
                continue
            seen.add(key)
            evidence.append(
                GameEvidence(
                    source_id=f"S{len(evidence)+1}",
                    game="stardew_valley",
                    evidence_type=str(item.get("entity_type") or "fact"),
                    source_catalog_id=key,
                    label=str(item.get("entity_name") or item.get("page_title") or key),
                    source_url=item.get("source_url"),
                    page_title=item.get("page_title"),
                    section_title=item.get("section_title"),
                    revision_id=item.get("revision_id"),
                    game_version=item.get("game_version"),
                    platform=item.get("platform"),
                    payload=dict(item),
                )
            )
        return evidence

    def _guide_retrieval(self, question: str, *, limit: int = 6) -> dict[str, Any]:
        if self.guide_store is None:
            return {
                "game": "stardew_valley", "status": "not_found", "intent": "guide",
                "query": question, "entity": None, "facts": {"hits": []},
                "warnings": ["Stardew guide database is not built."], "candidates": [], "provenance": [],
                "missing_context": [],
            }
        hits = self.guide_store.search(question, limit=limit)
        provenance = []
        for hit in hits:
            provenance.append(
                {
                    "game": "stardew_valley",
                    "entity_type": "guide_chunk",
                    "source_catalog_id": hit["chunk_id"],
                    "entity_name": hit["citation_label"],
                    "source_url": hit["source_url"],
                    "page_title": hit["page_title"],
                    "section_title": hit["section_title"],
                    "revision_id": hit["revision_id"],
                    "game_version": None,
                    "platform": "all",
                    "score": hit["score"],
                }
            )
        return {
            "game": "stardew_valley", "status": "found" if hits else "not_found",
            "intent": "guide", "query": question, "entity": None,
            "facts": {"hits": hits}, "warnings": [], "candidates": [],
            "provenance": provenance, "missing_context": [],
        }

    def answer(
        self,
        question: str,
        *,
        language: str = "auto",
        player_state: dict[str, Any] | None = None,
        include_debug: bool = False,
    ) -> GameGuideResult:
        if self._closed:
            raise RuntimeError("StardewAssistant is closed.")
        started = time.perf_counter()
        route = self.router.route(question, player_state=player_state)
        selected_language = route.language if language == "auto" else language
        if route.needs_context:
            retrieval = {
                "game": "stardew_valley", "status": "needs_context", "intent": route.intent,
                "query": question, "entity": route.entity, "facts": None,
                "warnings": ["The query is missing required player-state fields."],
                "candidates": [], "provenance": [], "missing_context": route.missing_context,
            }
        elif route.intent == "guide":
            retrieval = self._guide_retrieval(question)
        elif route.entity is None:
            retrieval = self.service.search(question)
        else:
            retrieval = self.service.query(
                route.intent,
                route.entity,
                player_state=route.player_state.to_dict(),
                require_current_state=(route.intent == "fish_availability" and any(term in question.casefold() for term in ("now", "today", "现在", "今天"))),
                bundle_mode=route.player_state.bundle_mode or "standard",
            )
        evidence = self._evidence_from_provenance(list(retrieval.get("provenance") or []))
        evidence_dicts = [item.to_dict() for item in evidence]
        if retrieval.get("facts") and isinstance(retrieval["facts"], dict) and isinstance(retrieval["facts"].get("hits"), list):
            by_chunk = {item.source_catalog_id: item.source_id for item in evidence}
            for hit in retrieval["facts"]["hits"]:
                hit["source_id"] = by_chunk.get(hit.get("chunk_id"))
        answer = self.renderer.render(retrieval, language=selected_language, evidence=evidence_dicts)
        context_payload = {
            "game": "stardew_valley",
            "question": question,
            "intent": retrieval.get("intent"),
            "entity": retrieval.get("entity"),
            "status": retrieval.get("status"),
            "player_state": route.player_state.to_dict(),
            "facts": retrieval.get("facts"),
            "warnings": retrieval.get("warnings") or [],
            "evidence": evidence_dicts,
        }
        debug = {
            "route": route.to_dict(),
            "total_time_seconds": round(time.perf_counter() - started, 6),
            "guide_database_available": self.guide_store is not None,
        } if include_debug else {}
        return GameGuideResult(
            game="stardew_valley",
            status=str(retrieval.get("status")),
            question=question,
            intent=str(retrieval.get("intent")),
            entity=retrieval.get("entity"),
            answer=answer,
            facts=retrieval.get("facts"),
            warnings=list(retrieval.get("warnings") or []),
            candidates=list(retrieval.get("candidates") or []),
            evidence=evidence,
            context_payload=context_payload,
            debug=debug,
        )
