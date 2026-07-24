"""Grounded natural-language assistant over the Terraria knowledge catalog."""

from .answer_validator import AnswerValidationResult, validate_grounded_answer
from .context_builder import ContextBuilder
from .document_retriever import DocumentRetriever
from .entity_aliases import resolve_entity_alias
from .entity_resolver import EntityResolver
from .generator import CallableAnswerGenerator, GroundedAnswerGenerator
from .hybrid_retriever import HybridRetriever
from .intent_router import IntentRouter
from .prompt_templates import build_grounded_messages
from .qwen_generator import QwenGroundedAnswerGenerator
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
    "AnswerValidationResult",
    "AssistantIntent",
    "AssistantRequest",
    "AssistantResponse",
    "ContextBundle",
    "ContextBuilder",
    "CallableAnswerGenerator",
    "GroundedAnswerGenerator",
    "QwenGroundedAnswerGenerator",
    "DeterministicAnswerRenderer",
    "DocumentRetriever",
    "EntityResolver",
    "HybridRetriever",
    "resolve_entity_alias",
    "IntentRouter",
    "RouteDecision",
    "StructuredRetriever",
    "TerrariaAssistant",
    "build_grounded_messages",
    "validate_grounded_answer",
]
