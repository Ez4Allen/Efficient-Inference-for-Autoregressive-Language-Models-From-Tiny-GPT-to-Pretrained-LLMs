# Stardew Valley guide corpus

This directory contains the tracked source manifest and attribution metadata for the Stardew Valley document-retrieval channel.

Build a small online smoke corpus:

```bash
python scripts/build_stardew_guides.py --max-pages 3
```

Build all configured pages:

```bash
python scripts/build_stardew_guides.py
```

Rebuild from an existing raw snapshot without network access:

```bash
python scripts/build_stardew_guides.py --offline
```
