# Terraria Guide Corpus

This directory is populated by `scripts/build_terraria_guides.py`.

Generated layers:

- `raw/pages.jsonl`: MediaWiki API responses and rendered page HTML.
- `cleaned/documents.jsonl`: section-aware, cleaned text documents.
- `chunks/chunks.jsonl`: deterministic retrieval chunks.
- `terraria_guides.sqlite3`: local SQLite FTS5 index.
- `reports/*.json`: import, cleaning, chunking, indexing, and build reports.

The tracked source manifest is `config/sources.yaml`. Large generated text and
SQLite files are ignored by Git by default. Attribution requirements are
recorded in `ATTRIBUTION.md` and in every generated record.

Build online:

```bash
python scripts/build_terraria_guides.py
```

Rebuild from an existing raw snapshot without network access:

```bash
python scripts/build_terraria_guides.py --offline
```

Package lightweight diagnostics for review:

```bash
python scripts/package_terraria_guide_diagnostics.py
```
