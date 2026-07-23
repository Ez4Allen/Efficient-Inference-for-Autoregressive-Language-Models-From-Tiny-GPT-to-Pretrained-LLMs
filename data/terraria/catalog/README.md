# Terraria Catalog

This directory contains the tracked cleaned snapshot, source attribution,
snapshot manifest, and lightweight build reports for the Terraria structured
knowledge pipeline.

## Tracked

- `cleaned/*.jsonl` — the reproducible input snapshot.
- `cleaned/*_report.json` — cleaning summaries.
- `snapshot_manifest.json` — cleaned-file SHA-256 hashes and expected counts.
- `linked/*_report.json` and `terraria_*_report.json` — lightweight reports.
- `ATTRIBUTION.md` — upstream source attribution.

## Generated locally

- `raw/*.jsonl`
- `normalized/*.jsonl`
- `linked/Recipes.jsonl`
- `linked/Drops.jsonl`
- `terraria_catalog.sqlite3`
- `terraria_query.sqlite3`

Build the derived layers from the tracked cleaned snapshot:

```bash
python scripts/build_terraria_knowledge.py --quiet
```

For a full refresh from the wiki Cargo API, run
`scripts/import_terraria_catalog.py`, rerun the cleaners, update the snapshot
manifest, and then build with `--no-strict-snapshot` during validation.
