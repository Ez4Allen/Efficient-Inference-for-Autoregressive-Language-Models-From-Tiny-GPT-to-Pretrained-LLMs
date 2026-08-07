# Stardew Valley structured release snapshot

The tracked course-release snapshot contains 505 versioned records:

```text
41 crops
55 fish
34 villagers
117 recipes
30 Standard Bundles
228 acquisition entities / 317 acquisition relations
```

Generate the source JSONL files and manifests:

```bash
python scripts/generate_stardew_release_data.py
```

Build the local SQLite database:

```bash
python scripts/build_stardew_knowledge.py --quiet
```

Validate the complete release contract:

```bash
python scripts/validate_stardew_release.py
```

Every record includes source URL, page/section metadata, game version, platform, license, parse status, and warnings. This is a defined course-project snapshot rather than a complete export of every Wiki page.
