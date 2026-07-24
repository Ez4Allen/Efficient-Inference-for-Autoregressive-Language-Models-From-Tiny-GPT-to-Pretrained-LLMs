from __future__ import annotations

from pathlib import Path

import pytest

from src.assistant import AssistantIntent, IntentRouter, TerrariaAssistant
from src.retrieval.guide_database import build_guide_database
from src.utils.io import write_jsonl


@pytest.fixture()
def guide_database(tmp_path: Path) -> Path:
    document = {
        "schema_version": 1,
        "document_id": "wiki:hardmode",
        "page_title": "Hardmode",
        "normalized_title": "hardmode",
        "page_id": 10,
        "revision_id": 100,
        "revision_timestamp": "2026-01-01T00:00:00Z",
        "source_url": "https://example.invalid/wiki/Hardmode",
        "language": "en",
        "source_name": "Official Terraria Wiki",
        "license_name": "CC BY-NC-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "retrieved_at": "2026-01-01T00:00:00Z",
        "quality_status": "revised",
        "quality_flags": ["revised"],
        "categories": ["Game mechanics"],
        "sections": [],
        "content_sha256": "a" * 64,
        "parse_status": "ok",
        "parse_warnings": [],
    }
    chunks = [
        {
            "schema_version": 1,
            "chunk_id": "wiki:hardmode::priorities::001",
            "document_id": "wiki:hardmode",
            "position": 1,
            "page_title": "Hardmode",
            "normalized_title": "hardmode",
            "section_id": "priorities",
            "section_title": "Early Hardmode priorities",
            "section_path": ["Early Hardmode priorities"],
            "text": (
                "After entering Hardmode, secure your base, improve mobility, "
                "monitor biome spread, and prepare for the mechanical bosses."
            ),
            "source_url": "https://example.invalid/wiki/Hardmode",
            "revision_id": 100,
            "language": "en",
            "quality_status": "revised",
            "quality_flags": ["revised"],
            "content_sha256": "b" * 64,
            "character_count": 130,
            "word_count": 19,
            "license_name": "CC BY-NC-SA 4.0",
            "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        },
        {
            "schema_version": 1,
            "chunk_id": "wiki:hardmode::spread::001",
            "document_id": "wiki:hardmode",
            "position": 2,
            "page_title": "Hardmode",
            "normalized_title": "hardmode",
            "section_id": "spread",
            "section_title": "Biome spread",
            "section_path": ["Early Hardmode priorities", "Biome spread"],
            "text": (
                "Contain important areas with tunnels and monitor the Corruption, "
                "Crimson, and Hallow as biome spread accelerates in Hardmode."
            ),
            "source_url": "https://example.invalid/wiki/Hardmode",
            "revision_id": 100,
            "language": "en",
            "quality_status": "revised",
            "quality_flags": ["revised"],
            "content_sha256": "c" * 64,
            "character_count": 137,
            "word_count": 20,
            "license_name": "CC BY-NC-SA 4.0",
            "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        },
    ]
    documents_path = tmp_path / "documents.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    database_path = tmp_path / "terraria_guides.sqlite3"
    write_jsonl(documents_path, [document])
    write_jsonl(chunks_path, chunks)
    build_guide_database(
        documents_path=documents_path,
        chunks_path=chunks_path,
        database_path=database_path,
        report_path=tmp_path / "index_report.json",
    )
    return database_path


@pytest.mark.parametrize(
    "question",
    [
        "What should I do after entering Hardmode?",
        "How should I prepare for Hardmode?",
        "进入困难模式后该做什么？",
        "怎么控制腐化扩散？",
    ],
)
def test_router_selects_guide_intent(question: str) -> None:
    decision = IntentRouter().route(question)
    assert decision.intent == AssistantIntent.GUIDE
    assert decision.confidence >= 0.8


def test_assistant_retrieves_guide_evidence(assistant_catalog, guide_database: Path) -> None:
    with TerrariaAssistant(
        assistant_catalog.database_path,
        guide_database_path=guide_database,
    ) as assistant:
        response = assistant.answer("What should I do after entering Hardmode?")

    assert response.status == "found"
    assert response.intent == AssistantIntent.GUIDE
    assert response.facts["hit_count"] >= 1
    assert "Hardmode" in response.answer
    assert "Source:" in response.answer
    assert any(row["entity_type"] == "guide_chunk" for row in response.evidence)
    assert response.context is not None
    assert "progression or mechanics" in response.context.text


def test_chinese_guide_query_uses_bilingual_expansion(
    assistant_catalog,
    guide_database: Path,
) -> None:
    with TerrariaAssistant(
        assistant_catalog.database_path,
        guide_database_path=guide_database,
    ) as assistant:
        response = assistant.answer("进入困难模式后该做什么？")

    assert response.status == "found"
    assert response.intent == AssistantIntent.GUIDE
    assert "本地 Terraria 攻略语料库" in response.answer
    assert response.facts["hits"][0]["page_title"] == "Hardmode"


def test_missing_guide_database_refuses_to_invent(
    assistant_catalog,
    tmp_path: Path,
) -> None:
    with TerrariaAssistant(
        assistant_catalog.database_path,
        guide_database_path=tmp_path / "missing.sqlite3",
    ) as assistant:
        response = assistant.answer("What should I do after entering Hardmode?")

    assert response.status == "not_found"
    assert "will not invent" in response.answer
    assert any("guide corpus is not built" in warning for warning in response.warnings)
