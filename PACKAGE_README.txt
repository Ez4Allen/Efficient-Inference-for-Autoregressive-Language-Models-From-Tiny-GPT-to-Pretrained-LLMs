GameGuideLM v1.0.0 release package

Validated release commands:
  python -m pytest -q
  python scripts/validate_release.py --skip-pytest

Verified offline result at packaging time:
  140 tests passed
  Terraria structured rebuild passed (14,353 resolved references)
  Stardew structured rebuild passed (31 starter facts)

Not included:
  Qwen model weights
  trained LoRA adapters
  generated SQLite databases
  online Wiki corpora
  benchmark result files

See README.md, RELEASE_NOTES.md, RELEASE_VALIDATION.json, and docs/DELIVERY.md.
