# Stardew Valley compact fact snapshot

This tracked starter snapshot contains 31 high-confidence records covering crops, fish, villagers, crafting recipes, and Standard Community Center bundles. It is intentionally compact: the database and services are designed so the teammate-maintained Stardew catalog can replace or extend it without changing the model runtime.

Build the local SQLite database with:

```bash
python scripts/build_stardew_knowledge.py --quiet
```
