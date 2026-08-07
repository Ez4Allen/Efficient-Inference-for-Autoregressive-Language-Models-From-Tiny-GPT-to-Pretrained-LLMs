# Stardew Valley guide corpus

The presentation build includes 25 compact project-authored offline guide pages. They produce 100 searchable sections/chunks and allow the demonstration to run without network access.

```bash
python scripts/build_stardew_guides.py --seed --quiet
```

Online source import remains supported:

```bash
python scripts/build_stardew_guides.py --max-pages 3
python scripts/build_stardew_guides.py
python scripts/build_stardew_guides.py --offline
```

The offline seed is explicitly marked `project_authored_summary`; it is not represented as verbatim Wiki content. Generated raw pages, cleaned documents, chunks, reports, and SQLite databases are build artifacts.
