# Data

Small reproducible datasets and the cleaned Terraria catalog snapshot are
tracked. Large generated artifacts, checkpoints, raw imports, linked JSONL,
and SQLite databases are excluded from Git.

- `tiny_shakespeare/`: tiny language-model demo corpus.
- `terraria/terraria_*_v1.jsonl`: supervised fine-tuning splits.
- `terraria/catalog/cleaned/`: tracked structured catalog snapshot.
- `terraria/catalog/snapshot_manifest.json`: hashes and expected counts for the tracked snapshot.

Rebuild the Terraria linked layer and query database with:

```bash
python scripts/build_terraria_knowledge.py --quiet
```
