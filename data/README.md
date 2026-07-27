# Data

GameGuideLM separates versioned compact snapshots from generated retrieval and
model artifacts.

## Versioned data

- `terraria/catalog/cleaned/`: cleaned Terraria Item, NPC, Recipe, and Drop
  snapshots used to rebuild the linked knowledge database.
- `terraria/catalog/snapshot_manifest.json`: expected counts and hashes for the
  Terraria snapshot.
- `terraria/terraria_*_v1.jsonl`: Terraria SFT and evaluation data.
- `stardew/catalog/cleaned/facts.jsonl`: compact, source-linked Stardew starter
  snapshot covering representative crops, fish, villagers, recipes, and
  Standard Bundles.
- `stardew/catalog/snapshot_manifest.json`: scope and version metadata for the
  Stardew starter snapshot.
- `stardew/evaluation/`: reviewed-format Stardew validation and evaluation
  seeds.
- `*/guides/config/sources.yaml`: source manifests for local Wiki guide builds.
- `sft_example.jsonl`: shared chat-style SFT format example.

## Generated local artifacts

The following are rebuilt locally and are intentionally excluded from Git:

- linked JSONL layers;
- raw/cleaned/chunked Wiki corpora;
- SQLite databases;
- diagnostic reports;
- evidence-conditioned SFT and teacher datasets;
- model checkpoints and evaluation outputs.

Rebuild the structured databases with:

```bash
python scripts/build_terraria_knowledge.py --quiet
python scripts/build_stardew_knowledge.py --quiet
```

Build the guide corpora with:

```bash
python scripts/build_terraria_guides.py
python scripts/build_stardew_guides.py
```

Wiki text remains subject to the attribution and licensing files stored in the
corresponding data directories.
