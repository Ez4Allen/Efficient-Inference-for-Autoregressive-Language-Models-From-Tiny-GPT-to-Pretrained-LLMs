"""End-to-end grounded Terraria assistant."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.knowledge.pipeline import build_terraria_knowledge
from src.knowledge.terraria_fact_service import TerrariaFactService
from src.knowledge.terraria_query_store import DEFAULT_DATABASE_PATH
from src.retrieval.guide_database import DEFAULT_GUIDE_DATABASE_PATH

from .context_builder import ContextBuilder
from .document_retriever import DocumentRetriever
from .generator import GroundedAnswerGenerator
from .entity_resolver import EntityResolver
from .hybrid_retriever import HybridRetriever
from .intent_router import IntentRouter
from .renderer import DeterministicAnswerRenderer
from .retriever import StructuredRetriever
from .schemas import AssistantRequest, AssistantResponse


class TerrariaAssistant:
    """Route, retrieve, ground, and render Terraria catalog answers."""

    def __init__(
        self,
        database_path: str | Path = DEFAULT_DATABASE_PATH,
        *,
        guide_database_path: str | Path = DEFAULT_GUIDE_DATABASE_PATH,
        auto_build: bool = False,
        router: IntentRouter | None = None,
        renderer: DeterministicAnswerRenderer | None = None,
        context_builder: ContextBuilder | None = None,
        generator: GroundedAnswerGenerator | None = None,
    ) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        if not self.database_path.exists():
            if not auto_build:
                raise FileNotFoundError(
                    "Terraria query database not found. Build it with "
                    "`python scripts/build_terraria_knowledge.py --quiet` or "
                    "construct TerrariaAssistant(auto_build=True). "
                    f"Expected path: {self.database_path}"
                )
            build_terraria_knowledge(
                database_path=self.database_path,
                strict_snapshot=True,
                verbose=False,
            )

        self.service = TerrariaFactService(self.database_path)
        self.guide_database_path = Path(guide_database_path).expanduser().resolve()
        self.router = router or IntentRouter()
        self.resolver = EntityResolver(self.service)
        self.structured_retriever = StructuredRetriever(self.service)
        self.document_retriever = DocumentRetriever(self.guide_database_path)
        self.retriever = HybridRetriever(
            self.structured_retriever,
            self.document_retriever,
        )
        self.renderer = renderer or DeterministicAnswerRenderer()
        self.context_builder = context_builder or ContextBuilder()
        self.generator = generator
        self._closed = False

    def __enter__(self) -> "TerrariaAssistant":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("TerrariaAssistant is closed.")

    def close(self) -> None:
        if not self._closed:
            self.document_retriever.close()
            self.service.close()
            if self.generator is not None and hasattr(self.generator, "close"):
                self.generator.close()
            self._closed = True

    def answer(
        self,
        question: str,
        *,
        mode: str = "normal",
        preferred_only: bool = True,
        include_partial: bool = True,
        language: str = "auto",
        guide_limit: int = 6,
        guide_minimum_score: float = 0.14,
        include_debug: bool = False,
    ) -> AssistantResponse:
        self._ensure_open()
        started = time.perf_counter()
        request = AssistantRequest(
            question=question,
            mode=mode,
            preferred_only=preferred_only,
            include_partial=include_partial,
            language=language,
        )
        selected_language = (
            self.router.detect_language(request.question)
            if request.language == "auto"
            else request.language
        )

        route_started = time.perf_counter()
        route = self.router.route(request)
        if route.intent.value == "guide":
            route.parameters["guide_limit"] = max(1, min(int(guide_limit), 20))
            route.parameters["guide_minimum_score"] = max(
                0.0,
                min(float(guide_minimum_score), 1.0),
            )
        route = self.resolver.resolve(route)
        route_seconds = time.perf_counter() - route_started

        if route.needs_clarification:
            retrieval: dict[str, Any] = {
                "status": "ambiguous",
                "intent": route.intent.value,
                "query": route.entity,
                "facts": None,
                "candidates": route.candidates,
                "warnings": [route.clarification_question],
                "provenance": [
                    {
                        "entity_type": candidate.get("entity_type"),
                        "source_catalog_id": candidate.get("source_catalog_id"),
                    }
                    for candidate in route.candidates
                    if candidate.get("source_catalog_id")
                ],
            }
            retrieval_seconds = 0.0
        else:
            retrieval_started = time.perf_counter()
            retrieval = self.retriever.retrieve(request, route)
            retrieval_seconds = time.perf_counter() - retrieval_started

        context_started = time.perf_counter()
        context = self.context_builder.build(
            request,
            route,
            retrieval,
            language=selected_language,
        )
        context_seconds = time.perf_counter() - context_started

        render_started = time.perf_counter()
        fallback_answer = self.renderer.render(
            request,
            route,
            retrieval,
            language=selected_language,
        )
        generator_error: str | None = None
        if self.generator is None:
            answer = fallback_answer
        else:
            try:
                answer = self.generator.generate(context, fallback_answer)
            except Exception as error:
                generator_error = f"{type(error).__name__}: {error}"
                answer = fallback_answer
        render_seconds = time.perf_counter() - render_started

        generator_warnings = (
            list(getattr(self.generator, "last_warnings", []) or [])
            if self.generator is not None
            else []
        )
        if generator_error:
            generator_warnings.append(
                "LLM generation raised an error; the deterministic grounded answer was used."
            )
        warnings = list(dict.fromkeys(
            [
                warning
                for warning in (
                    list(retrieval.get("warnings") or []) + generator_warnings
                )
                if warning
            ]
        ))
        status = "clarification" if route.needs_clarification else str(retrieval.get("status", "unknown"))
        debug = {
            "language": selected_language,
            "timings_seconds": {
                "routing_and_resolution": round(route_seconds, 6),
                "retrieval": round(retrieval_seconds, 6),
                "context": round(context_seconds, 6),
                "render_or_generation": round(render_seconds, 6),
                "total": round(time.perf_counter() - started, 6),
            },
        }
        if self.generator is not None:
            debug["generation"] = dict(
                getattr(self.generator, "last_debug", {}) or {}
            )
        if generator_error:
            debug["generation_error"] = generator_error
        if not include_debug:
            debug = {}

        return AssistantResponse(
            status=status,
            question=request.question,
            answer=answer,
            intent=route.intent,
            entity=route.entity,
            facts=retrieval.get("facts"),
            warnings=warnings,
            candidates=list(retrieval.get("candidates") or route.candidates),
            evidence=list(context.evidence),
            route=route,
            context=context,
            debug=debug,
        )

    def answer_dict(self, question: str, **kwargs: Any) -> dict[str, Any]:
        include_debug = bool(kwargs.get("include_debug", False))
        return self.answer(question, **kwargs).to_dict(include_debug=include_debug)

    def context_json(self, question: str, **kwargs: Any) -> str:
        response = self.answer(question, **kwargs)
        if response.context is None:
            return "{}"
        return json.dumps(response.context.to_dict(), ensure_ascii=False, indent=2)
