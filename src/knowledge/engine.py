
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.knowledge.query_service import StructuredFactQueryService
from src.knowledge.renderer import render_query_result
from src.knowledge.store import StructuredKnowledgeStore


class StructuredKnowledgeEngine:
    """
    High-level entry point for structured Terraria facts.

    The knowledge file is loaded once when the engine is created.
    After that, repeated queries use the in-memory indexes.
    """

    def __init__(
        self,
        store: StructuredKnowledgeStore,
    ) -> None:
        self.store = store
        self.query_service = StructuredFactQueryService(
            store
        )

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
    ) -> "StructuredKnowledgeEngine":
        """
        Create an engine from one JSONL knowledge file.
        """
        store = StructuredKnowledgeStore.from_jsonl(
            path
        )

        return cls(store)


    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
    ) -> "StructuredKnowledgeEngine":
        """
        Create an engine by loading all JSONL files
        inside one knowledge directory.
        """
        store = StructuredKnowledgeStore.from_directory(
            directory
        )

        return cls(store)

    def query(
        self,
        entity_name: str,
        fact_type: str | None = None,
        require_verified: bool = False,
    ) -> dict[str, Any]:
        """
        Return the structured query result without rendering it.
        """
        return self.query_service.query(
            entity_name=entity_name,
            fact_type=fact_type,
            require_verified=require_verified,
        )

    def answer(
        self,
        entity_name: str,
        fact_type: str | None = None,
        require_verified: bool = False,
    ) -> dict[str, Any]:
        """
        Query the database and return both:

        - result: structured machine-readable data
        - text: deterministic human-readable answer
        """
        result = self.query(
            entity_name=entity_name,
            fact_type=fact_type,
            require_verified=require_verified,
        )

        text = render_query_result(result)

        return {
            "result": result,
            "text": text,
        }

    def __len__(self) -> int:
        """
        Return the number of structured records loaded.
        """
        return len(self.store)
