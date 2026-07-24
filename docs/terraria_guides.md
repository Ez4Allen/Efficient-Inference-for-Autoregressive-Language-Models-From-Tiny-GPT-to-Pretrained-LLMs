# Terraria Guide Corpus and Document Retriever

## Purpose

The structured Terraria catalog is authoritative for Item, NPC, Recipe, and
Drop records. It is not sufficient for open-ended questions such as:

- What should I do on the first night?
- What should I prioritize after entering Hardmode?
- How should I prepare for Plantera?
- How do I control biome spread?
- Which class setup should I use at a progression stage?

The guide corpus supplies grounded excerpts for these progression, strategy,
and mechanics questions.

## Source and license

The default source is the Official Terraria Wiki. The importer discovers
current pages from `Category:Guides`, excludes `Legacy:` pages by default, and
adds explicitly configured core mechanics pages. Quality categories are
recorded so revised, under-revision, and subject-to-revision pages remain
visible to retrieval and answer generation.

Text that the Official Terraria Wiki may lawfully license is provided under
CC BY-NC-SA 4.0 unless otherwise noted. The pipeline imports text only and
preserves page URL, revision ID, retrieval timestamp, quality flags, and
license metadata in every document and chunk. See
`data/terraria/guides/ATTRIBUTION.md`.

## Build

```bash
python scripts/build_terraria_guides.py
```

Useful development modes:

```bash
# Fetch only a few pages for a network/cleaner smoke test.
python scripts/build_terraria_guides.py --max-pages 5

# Force a complete page refresh.
python scripts/build_terraria_guides.py --refresh

# Re-clean, re-chunk, and re-index an existing raw snapshot offline.
python scripts/build_terraria_guides.py --offline
```

The build stages are:

1. category and explicit-page discovery;
2. rate-limited MediaWiki API import;
3. section-aware HTML cleaning;
4. paragraph-aware overlapping chunking;
5. static quality audit;
6. SQLite FTS5 index construction.

## Data layout

```text
data/terraria/guides/
├── config/sources.yaml
├── raw/pages.jsonl
├── cleaned/documents.jsonl
├── chunks/chunks.jsonl
├── reports/
├── ATTRIBUTION.md
└── terraria_guides.sqlite3
```

Generated raw, cleaned, chunk, report, and SQLite files are ignored by Git.
The source manifest, attribution, and documentation are tracked.

## Retrieval

```python
from src.retrieval import GuideDocumentStore

with GuideDocumentStore() as store:
    hits = store.search(
        "What should I do after entering Hardmode?",
        limit=6,
    )
```

The retriever uses SQLite FTS5 plus deterministic English/Chinese query
expansion and Python-side reranking. It does not require an embedding model or
external vector database.

## Assistant integration

`TerrariaAssistant` dispatches structured catalog questions to
`TerrariaFactService` and progression/mechanics questions to the guide corpus.

```python
from src.assistant import TerrariaAssistant

with TerrariaAssistant() as assistant:
    response = assistant.answer(
        "进入困难模式后该做什么？"
    )
    print(response.answer)
    print(response.evidence)
```

When no guide database exists, or no chunk exceeds the retrieval threshold,
the Assistant refuses to invent progression advice and explains how to build
the local corpus.

## Quality review

A live Wiki crawl can reveal page-specific markup that synthetic tests do not
cover. Package a bounded diagnostic bundle after the first full build:

```bash
python scripts/package_terraria_guide_diagnostics.py
```

The resulting ZIP contains:

- source manifest;
- import, cleaning, chunking, quality, index, and build reports;
- evenly sampled cleaned documents;
- suspicious/under-revision document samples;
- evenly sampled chunks.

It excludes full raw HTML, the complete corpus, and SQLite databases. This is
the preferred artifact for external cleaning review.

## Stage-aware reranking

Broad guide questions need more than lexical similarity. The local retriever
assigns deterministic query profiles for common tasks such as first-night
survival, early Hardmode priorities, boss progression, and biome-spread
control. Profiles add title and section-path preferences, reject incompatible
stages (for example, `Pre-Hardmode` evidence for a question about actions after
entering Hardmode), and diversify the final evidence across independent
sections.

This reranking remains fully inspectable: no embedding model or hidden LLM
classification is used. Chinese progression questions are expanded into the
same English stage anchors used by the Wiki corpus, including `Early
Hardmode`, `after Wall of Flesh`, ores, biome spread, and mechanical-boss
preparation.
