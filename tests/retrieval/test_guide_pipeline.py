from __future__ import annotations

import json
from pathlib import Path

from src.retrieval.guide_database import GuideDocumentStore, build_guide_database
from src.retrieval.text_chunker import chunk_guide_documents
from src.retrieval.wiki_cleaner import clean_wiki_pages
from src.retrieval.wiki_importer import discover_pages, import_wiki_pages
from src.utils.io import write_jsonl, write_yaml


class FakeWikiClient:
    def category_members(self, category: str) -> list[str]:
        return {
            "Category:Guides": [
                "Guide:Getting started",
                "Guide:Game progression",
                "Legacy:Guide:Walkthrough",
                "Terraria Wiki:Projects/Guides",
            ],
            "Category:Revised Guides": ["Guide:Getting started"],
            "Category:Guides under revision": ["Guide:Game progression"],
            "Category:Guides subject to revision": [],
        }[category]


def manifest() -> dict:
    return {
        "version": 1,
        "wiki": {
            "api_url": "https://example.invalid/api.php",
            "article_base_url": "https://example.invalid/wiki/",
            "language": "en",
        },
        "discovery": {
            "guide_categories": ["Category:Guides"],
            "quality_categories": {
                "revised": "Category:Revised Guides",
                "under_revision": "Category:Guides under revision",
                "subject_to_revision": "Category:Guides subject to revision",
            },
            "include_title_prefixes": ["Guide:"],
            "exclude_title_prefixes": ["Legacy:", "Terraria Wiki:"],
            "exclude_titles": [],
            "explicit_pages": ["Hardmode"],
        },
        "cleaning": {
            "excluded_sections": ["References", "History"],
            "minimum_document_characters": 100,
            "minimum_section_characters": 20,
        },
        "chunking": {
            "maximum_characters": 500,
            "overlap_characters": 50,
            "minimum_characters": 80,
        },
    }


def test_discovery_filters_legacy_and_marks_quality() -> None:
    pages = discover_pages(FakeWikiClient(), manifest())
    assert [page.title for page in pages] == [
        "Guide:Game progression",
        "Guide:Getting started",
        "Hardmode",
    ]
    by_title = {page.title: page for page in pages}
    assert by_title["Guide:Getting started"].quality_status == "revised"
    assert by_title["Guide:Game progression"].quality_status == "under_revision"
    assert by_title["Hardmode"].source_kind == "explicit_page"


def _raw_records() -> list[dict]:
    return [
        {
            "schema_version": 1,
            "requested_title": "Guide:Getting started",
            "title": "Guide:Getting started",
            "page_id": 10,
            "revision_id": 101,
            "revision_timestamp": "2026-01-01T00:00:00Z",
            "source_url": "https://example.invalid/wiki/Guide:Getting_started",
            "html": """
                <div class="mw-parser-output">
                  <table class="ambox"><tr><td>Navigation warning</td></tr></table>
                  <p>Begin by gathering wood and crafting basic tools. Build a shelter before night.</p>
                  <h2><span class="mw-headline">First night</span></h2>
                  <p>Place walls, a light source, a table, and a chair inside a valid room.</p>
                  <ul><li>Craft torches from gel and wood.</li><li>Avoid fighting large groups.</li></ul>
                  <h2>References</h2><p>This should be excluded.</p>
                </div>
            """,
            "categories": ["Guides", "Revised Guides"],
            "quality_status": "revised",
            "quality_flags": ["revised"],
            "language": "en",
            "source_name": "Official Terraria Wiki",
            "license": {"name": "CC BY-NC-SA 4.0", "url": "https://creativecommons.org/licenses/by-nc-sa/4.0/"},
            "fetched_at": "2026-01-01T00:00:00+00:00",
        },
        {
            "schema_version": 1,
            "requested_title": "Hardmode",
            "title": "Hardmode",
            "page_id": 11,
            "revision_id": 102,
            "revision_timestamp": "2026-01-01T00:00:00Z",
            "source_url": "https://example.invalid/wiki/Hardmode",
            "html": """
                <div class="mw-parser-output">
                  <p>Hardmode begins after defeating the Wall of Flesh and introduces stronger enemies.</p>
                  <h2>Early Hardmode priorities</h2>
                  <p>Secure your base, review biome spread, and improve mobility before fighting mechanical bosses.</p>
                  <h3>Biome spread</h3>
                  <p>Plan containment tunnels and monitor the Corruption, Crimson, and Hallow around important builds.</p>
                </div>
            """,
            "categories": ["Game mechanics"],
            "quality_status": "unknown",
            "quality_flags": [],
            "language": "en",
            "source_name": "Official Terraria Wiki",
            "license": {"name": "CC BY-NC-SA 4.0", "url": "https://creativecommons.org/licenses/by-nc-sa/4.0/"},
            "fetched_at": "2026-01-01T00:00:00+00:00",
        },
    ]


def test_clean_chunk_index_and_search(tmp_path: Path) -> None:
    root = tmp_path / "guides"
    manifest_path = root / "config" / "sources.yaml"
    raw_path = root / "raw" / "pages.jsonl"
    documents_path = root / "cleaned" / "documents.jsonl"
    chunks_path = root / "chunks" / "chunks.jsonl"
    database_path = root / "terraria_guides.sqlite3"
    write_yaml(manifest_path, manifest())
    write_jsonl(raw_path, _raw_records())

    cleaning_report = clean_wiki_pages(
        input_path=raw_path,
        output_path=documents_path,
        report_path=root / "reports" / "cleaning_report.json",
        manifest_path=manifest_path,
    )
    assert cleaning_report["written_documents"] == 2
    documents = [json.loads(line) for line in documents_path.read_text().splitlines()]
    serialized_documents = json.dumps(documents)
    assert "This should be excluded" not in serialized_documents
    assert "Navigation warning" not in serialized_documents
    assert any(section["title"] == "First night" for section in documents[0]["sections"])

    chunk_report = chunk_guide_documents(
        input_path=documents_path,
        output_path=chunks_path,
        report_path=root / "reports" / "chunking_report.json",
        manifest_path=manifest_path,
    )
    assert chunk_report["written_chunks"] >= 3

    index_report = build_guide_database(
        documents_path=documents_path,
        chunks_path=chunks_path,
        database_path=database_path,
        report_path=root / "reports" / "index_report.json",
    )
    assert index_report["status"] == "passed"

    with GuideDocumentStore(database_path) as store:
        first_night = store.search("What should I do on the first night?", limit=3)
        assert first_night
        assert first_night[0]["page_title"] == "Guide:Getting started"
        assert "shelter" in first_night[0]["text"].casefold() or "night" in first_night[0]["section_title"].casefold()

        chinese = store.search("进入困难模式后怎么发展？", limit=3)
        assert chinese
        assert chinese[0]["page_title"] == "Hardmode"
        assert chinese[0]["source_url"].startswith("https://")


class FakeImportClient(FakeWikiClient):
    def page_revision_metadata(self, titles: list[str]) -> dict[str, dict]:
        return {
            title: {
                "requested_title": title,
                "title": title,
                "page_id": index,
                "revision_id": 1000 + index,
                "revision_timestamp": "2026-01-01T00:00:00Z",
                "source_url": f"https://example.invalid/wiki/{title.replace(' ', '_')}",
                "missing": False,
            }
            for index, title in enumerate(titles, start=1)
        }

    def parse_page(self, title: str) -> dict:
        revision_ids = {
            "Guide:Game progression": 1001,
            "Guide:Getting started": 1002,
            "Hardmode": 1003,
        }
        return {
            "requested_title": title,
            "title": title,
            "page_id": revision_ids[title] - 1000,
            "revision_id": revision_ids[title],
            "display_title": title,
            "source_url": f"https://example.invalid/wiki/{title.replace(' ', '_')}",
            "html": (
                "<div class='mw-parser-output'><p>"
                + ("Useful guide text " * 20)
                + "</p></div>"
            ),
            "sections_api": [],
            "categories": ["Guides"],
            "properties": [],
        }

    def close(self) -> None:
        pass


def test_importer_writes_incremental_raw_snapshot(tmp_path: Path) -> None:
    manifest_path = tmp_path / "sources.yaml"
    output_path = tmp_path / "pages.jsonl"
    report_path = tmp_path / "import_report.json"
    write_yaml(manifest_path, manifest())
    report = import_wiki_pages(
        manifest_path=manifest_path,
        output_path=output_path,
        report_path=report_path,
        client=FakeImportClient(),
        verbose=False,
    )
    assert report["status"] == "passed"
    assert report["written_pages"] == 3
    assert report["fetched_pages"] == 3
    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert all(record["license"] == {} for record in records)
    assert all(len(record["content_sha256"]) == 64 for record in records)

    second = import_wiki_pages(
        manifest_path=manifest_path,
        output_path=output_path,
        report_path=report_path,
        client=FakeImportClient(),
        verbose=False,
    )
    assert second["reused_pages"] == 3
    assert second["fetched_pages"] == 0


def test_discovery_priority_controls_smoke_sample() -> None:
    configured = manifest()
    configured["discovery"]["priority_pages"] = [
        "Guide:Getting started",
        "Hardmode",
    ]
    configured["discovery"]["reference_pages"] = ["Hardmode"]
    pages = discover_pages(FakeWikiClient(), configured, max_pages=2)
    assert [page.title for page in pages] == [
        "Guide:Getting started",
        "Hardmode",
    ]
    assert pages[0].retrieval_role == "guide"
    assert pages[1].retrieval_role == "reference"


def test_advice_query_prefers_narrative_guide_over_reference_table(tmp_path: Path) -> None:
    documents_path = tmp_path / "documents.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    database_path = tmp_path / "guides.sqlite3"
    base_document = {
        "schema_version": 1,
        "page_id": 1,
        "revision_id": 1,
        "revision_timestamp": "2026-01-01T00:00:00Z",
        "language": "en",
        "source_name": "Official Terraria Wiki",
        "license_name": "CC BY-NC-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "retrieved_at": "2026-01-01T00:00:00Z",
        "quality_status": "revised",
        "quality_flags": ["revised"],
        "categories": [],
        "sections": [],
        "parse_status": "ok",
        "parse_warnings": [],
        "source_kind": "explicit_page",
        "discovery_priority": 1,
    }
    documents = [
        {
            **base_document,
            "document_id": "wiki:hardmode",
            "page_title": "Hardmode",
            "normalized_title": "hardmode",
            "source_url": "https://example.invalid/wiki/Hardmode",
            "content_sha256": "a" * 64,
            "retrieval_role": "mechanics",
        },
        {
            **base_document,
            "document_id": "wiki:armor",
            "page_title": "Armor",
            "normalized_title": "armor",
            "source_url": "https://example.invalid/wiki/Armor",
            "content_sha256": "b" * 64,
            "retrieval_role": "reference",
        },
    ]
    common_chunk = {
        "schema_version": 2,
        "position": 1,
        "revision_id": 1,
        "language": "en",
        "quality_status": "revised",
        "quality_flags": ["revised"],
        "source_kind": "explicit_page",
        "discovery_priority": 1,
        "character_count": 200,
        "word_count": 30,
    }
    chunks = [
        {
            **common_chunk,
            "chunk_id": "wiki:hardmode::early::001",
            "document_id": "wiki:hardmode",
            "page_title": "Hardmode",
            "normalized_title": "hardmode",
            "section_id": "early",
            "section_title": "Early Hardmode priorities",
            "section_path": ["Early Hardmode priorities"],
            "text": "After entering Hardmode, improve mobility, protect the base, monitor biome spread, and prepare for mechanical bosses.",
            "source_url": "https://example.invalid/wiki/Hardmode",
            "retrieval_role": "mechanics",
            "content_kind": "prose",
            "table_row_count": 0,
            "table_density": 0.0,
            "content_sha256": "c" * 64,
        },
        {
            **common_chunk,
            "chunk_id": "wiki:armor::hardmode::001",
            "document_id": "wiki:armor",
            "page_title": "Armor",
            "normalized_title": "armor",
            "section_id": "hardmode",
            "section_title": "Hardmode",
            "section_path": ["Armor sets", "Hardmode"],
            "text": "Cobalt armor | Hardmode | defense | progression | Mythril armor | Hardmode | defense | preparation",
            "source_url": "https://example.invalid/wiki/Armor",
            "retrieval_role": "reference",
            "content_kind": "table",
            "table_row_count": 2,
            "table_density": 1.0,
            "content_sha256": "d" * 64,
        },
    ]
    write_jsonl(documents_path, documents)
    write_jsonl(chunks_path, chunks)
    build_guide_database(
        documents_path=documents_path,
        chunks_path=chunks_path,
        database_path=database_path,
        report_path=tmp_path / "report.json",
    )
    with GuideDocumentStore(database_path) as store:
        hits = store.search("What should I do after entering Hardmode?", limit=2)
    assert hits
    assert hits[0]["page_title"] == "Hardmode"
    assert hits[0]["retrieval_role"] == "mechanics"
    assert hits[0]["content_kind"] == "prose"


def _stage_ranking_fixture(tmp_path: Path) -> Path:
    documents_path = tmp_path / "stage_documents.jsonl"
    chunks_path = tmp_path / "stage_chunks.jsonl"
    database_path = tmp_path / "stage_guides.sqlite3"
    base_document = {
        "schema_version": 2,
        "page_id": 1,
        "revision_id": 1,
        "revision_timestamp": "2026-01-01T00:00:00Z",
        "language": "en",
        "source_name": "Official Terraria Wiki",
        "license_name": "CC BY-NC-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "retrieved_at": "2026-01-01T00:00:00Z",
        "quality_status": "revised",
        "quality_flags": ["revised"],
        "categories": [],
        "sections": [],
        "parse_status": "ok",
        "parse_warnings": [],
        "source_kind": "explicit_page",
        "retrieval_role": "guide",
        "discovery_priority": 0,
    }
    documents = [
        {
            **base_document,
            "document_id": "wiki:progression",
            "page_title": "Guide:Game progression",
            "normalized_title": "guidegameprogression",
            "source_url": "https://example.invalid/wiki/Guide:Game_progression",
            "content_sha256": "a" * 64,
        },
        {
            **base_document,
            "document_id": "wiki:hardmode",
            "page_title": "Hardmode",
            "normalized_title": "hardmode",
            "source_url": "https://example.invalid/wiki/Hardmode",
            "content_sha256": "b" * 64,
            "retrieval_role": "mechanics",
        },
    ]
    common_chunk = {
        "schema_version": 2,
        "position": 1,
        "revision_id": 1,
        "language": "en",
        "quality_status": "revised",
        "quality_flags": ["revised"],
        "source_kind": "explicit_page",
        "discovery_priority": 0,
        "content_kind": "prose",
        "table_row_count": 0,
        "table_density": 0.0,
        "character_count": 240,
        "word_count": 36,
        "license_name": "CC BY-NC-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    }
    chunks = [
        {
            **common_chunk,
            "chunk_id": "wiki:progression::overview::001",
            "document_id": "wiki:progression",
            "page_title": "Guide:Game progression",
            "normalized_title": "guidegameprogression",
            "section_id": "overview",
            "section_title": "Overview",
            "section_path": ["Overview"],
            "text": (
                "Several bosses are significant for progression: Eye of Cthulhu, "
                "Skeletron, Wall of Flesh, the mechanical bosses, Plantera, Golem, "
                "Lunatic Cultist, and Moon Lord form the main progression path."
            ),
            "source_url": "https://example.invalid/wiki/Guide:Game_progression",
            "retrieval_role": "guide",
            "content_sha256": "c" * 64,
        },
        {
            **common_chunk,
            "chunk_id": "wiki:progression::early-hardmode::001",
            "document_id": "wiki:progression",
            "page_title": "Guide:Game progression",
            "normalized_title": "guidegameprogression",
            "section_id": "early-hardmode",
            "section_title": "Early Hardmode",
            "section_path": ["Hardmode", "Early Hardmode", "Tips"],
            "text": (
                "After defeating the Wall of Flesh and entering Hardmode, secure "
                "important areas, obtain early Hardmode ores, monitor biome spread, "
                "and prepare for the mechanical bosses."
            ),
            "source_url": "https://example.invalid/wiki/Guide:Game_progression",
            "retrieval_role": "guide",
            "content_sha256": "d" * 64,
        },
        {
            **common_chunk,
            "chunk_id": "wiki:progression::pre-hardmode::001",
            "document_id": "wiki:progression",
            "page_title": "Guide:Game progression",
            "normalized_title": "guidegameprogression",
            "section_id": "underworld",
            "section_title": "The Underworld",
            "section_path": ["Pre-Hardmode", "The Underworld"],
            "text": (
                "Prepare for Hardmode before fighting the Wall of Flesh by building "
                "a bridge and gathering pre-Hardmode equipment."
            ),
            "source_url": "https://example.invalid/wiki/Guide:Game_progression",
            "retrieval_role": "guide",
            "content_sha256": "e" * 64,
        },
        {
            **common_chunk,
            "chunk_id": "wiki:progression::after-plantera::001",
            "document_id": "wiki:progression",
            "page_title": "Guide:Game progression",
            "normalized_title": "guidegameprogression",
            "section_id": "after-plantera",
            "section_title": "After Plantera",
            "section_path": ["Hardmode", "After Plantera"],
            "text": (
                "After Plantera, the optional Empress of Light is often recommended "
                "after Golem and can reward several powerful weapons."
            ),
            "source_url": "https://example.invalid/wiki/Guide:Game_progression",
            "retrieval_role": "guide",
            "content_sha256": "f" * 64,
        },
        {
            **common_chunk,
            "chunk_id": "wiki:progression::hardmode-jungle::001",
            "document_id": "wiki:progression",
            "page_title": "Guide:Game progression",
            "normalized_title": "guidegameprogression",
            "section_id": "hardmode-jungle",
            "section_title": "Hardmode Jungle",
            "section_path": ["Hardmode", "Hardmode Jungle"],
            "text": "Defeating all three mechanical bosses allows Chlorophyte progression in the Jungle.",
            "source_url": "https://example.invalid/wiki/Guide:Game_progression",
            "retrieval_role": "guide",
            "content_sha256": "1" * 64,
        },
        {
            **common_chunk,
            "chunk_id": "wiki:hardmode::spread::001",
            "document_id": "wiki:hardmode",
            "page_title": "Hardmode",
            "normalized_title": "hardmode",
            "section_id": "spread",
            "section_title": "Spread",
            "section_path": ["Biomes", "Spread"],
            "text": (
                "In Hardmode, Corruption, Crimson, and Hallow spread through nearby "
                "blocks. Containment tunnels can protect important builds."
            ),
            "source_url": "https://example.invalid/wiki/Hardmode",
            "retrieval_role": "mechanics",
            "content_sha256": "2" * 64,
        },
        {
            **common_chunk,
            "chunk_id": "wiki:hardmode::empress::001",
            "document_id": "wiki:hardmode",
            "page_title": "Hardmode",
            "normalized_title": "hardmode",
            "section_id": "empress",
            "section_title": "Empress of Light",
            "section_path": ["Bosses", "Empress of Light"],
            "text": (
                "The Empress of Light appears in the Hallow and is an optional "
                "Hardmode boss with powerful attacks."
            ),
            "source_url": "https://example.invalid/wiki/Hardmode",
            "retrieval_role": "mechanics",
            "content_sha256": "3" * 64,
        },
    ]
    write_jsonl(documents_path, documents)
    write_jsonl(chunks_path, chunks)
    build_guide_database(
        documents_path=documents_path,
        chunks_path=chunks_path,
        database_path=database_path,
        report_path=tmp_path / "stage_report.json",
    )
    return database_path


def test_chinese_early_hardmode_filters_pre_hardmode_and_prefers_priorities(
    tmp_path: Path,
) -> None:
    database_path = _stage_ranking_fixture(tmp_path)
    with GuideDocumentStore(database_path) as store:
        hits = store.search("进入困难模式后该做什么？", limit=5)

    assert hits
    assert hits[0]["section_title"] == "Early Hardmode"
    assert all("Pre-Hardmode" not in hit["section_path"] for hit in hits)
    assert hits[0]["score"] > hits[-1]["score"]


def test_boss_progression_prefers_cross_game_overview_over_optional_boss(
    tmp_path: Path,
) -> None:
    database_path = _stage_ranking_fixture(tmp_path)
    with GuideDocumentStore(database_path) as store:
        hits = store.search("What is the recommended boss progression?", limit=10)

    assert hits
    assert hits[0]["page_title"] == "Guide:Game progression"
    assert hits[0]["section_title"] == "Overview"
    assert all(hit["section_title"] != "After Plantera" for hit in hits)


def test_biome_spread_profile_rejects_unrelated_hardmode_bosses(
    tmp_path: Path,
) -> None:
    database_path = _stage_ranking_fixture(tmp_path)
    with GuideDocumentStore(database_path) as store:
        hits = store.search("How do I control biome spread?", limit=6)

    assert hits
    assert hits[0]["section_title"] == "Spread"
    assert all(hit["section_title"] != "Empress of Light" for hit in hits)
