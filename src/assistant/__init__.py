"""Grounded natural-language assistant over the Terraria knowledge catalog."""

from .context_builder import ContextBuilder
from .entity_aliases import resolve_entity_alias
from .entity_resolver import EntityResolver
from .generator import CallableAnswerGenerator, GroundedAnswerGenerator
from .intent_router import IntentRouter
from .renderer import DeterministicAnswerRenderer
from .retriever import StructuredRetriever
from .schemas import (
    AssistantIntent,
    AssistantRequest,
    AssistantResponse,
    ContextBundle,
    RouteDecision,
)
from .terraria_assistant import TerrariaAssistant

__all__ = [
    "AssistantIntent",
    "AssistantRequest",
    "AssistantResponse",
    "ContextBundle",
    "ContextBuilder",
    "CallableAnswerGenerator",
    "GroundedAnswerGenerator",
    "DeterministicAnswerRenderer",
    "EntityResolver",
    "resolve_entity_alias",
    "IntentRouter",
    "RouteDecision",
    "StructuredRetriever",
    "TerrariaAssistant",
]
