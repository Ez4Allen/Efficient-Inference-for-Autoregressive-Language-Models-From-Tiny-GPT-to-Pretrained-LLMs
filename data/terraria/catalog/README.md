# Terraria Catalog

This directory contains reports and attribution metadata for the
Terraria structured knowledge pipeline.

The large catalog data layers are generated locally and are not stored
in Git:

- `raw/*.jsonl`
- `normalized/*.jsonl`
- `cleaned/*.jsonl`
- `linked/*.jsonl`
- `terraria_catalog.sqlite3`
- `terraria_query.sqlite3`

The pipeline is reproducible from the tracked Python modules:

1. `scripts/import_terraria_catalog.py`
2. `src/knowledge/cleaning/`
3. `src/knowledge/linking/`
4. `src/knowledge/catalog_database_builder.py`

Tracked JSON reports preserve record counts, integrity results, linkage
coverage and source SHA-256 values.

The small manually maintained facts under
`data/terraria/structured/` are tracked in Git.
