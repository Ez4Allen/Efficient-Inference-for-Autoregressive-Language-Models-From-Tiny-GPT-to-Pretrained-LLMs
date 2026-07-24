"""Hybrid structured/document retrieval dispatch for the Terraria assistant."""

from __future__ import annotations

from typing import Any

from .document_retriever import DocumentRetriever
from .retriever import StructuredRetriever
from .schemas import AssistantIntent, AssistantRequest, RouteDecision


class HybridRetriever:
    """Use FactService for catalog intents and guide FTS for advice intents."""

    def __init__(
        self,
        structured: StructuredRetriever,
        documents: DocumentRetriever,
    ) -> None:
        self.structured = structured
        self.documents = documents

    def retrieve(
        self,
        request: AssistantRequest,
        route: RouteDecision,
    ) -> dict[str, Any]:
        if route.intent == AssistantIntent.GUIDE:
            return self.documents.retrieve(request, route)
        return self.structured.retrieve(request, route)
