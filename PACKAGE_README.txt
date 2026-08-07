GameGuideLM v1.1.0 - code and data release

Primary offline build:
  python scripts/build_stardew_release.py

Validated release commands:
  python -m pytest -q
  python scripts/validate_stardew_release.py
  python scripts/validate_release.py --skip-pytest

Verified offline result for this rebased patch:
  171 tests passed
  Stardew structured catalog: 505 records
  Stardew acquisition relations: 317
  Stardew offline guides: 25 pages / 100 searchable chunks
  Stardew deterministic regression: 100/100 cases passed
  English / Chinese regression split: 50 / 50
  Grounded Stardew training records: 159 train / 17 validation
  Legacy Stardew SFT candidates: 1,262 audited, all pending human review
  Terraria structured rebuild: 14,353 resolved references

Included demonstration assets:
  demo/stardew_showcase.html
  notebooks/04_stardew_release_demo.ipynb

Intentionally not included yet:
  final written report
  PPTX/PDF presentation
  Qwen model weights
  trained LoRA adapters
  trained TinyQwenDraft checkpoint
  generated SQLite databases and runtime result logs

Truth boundary:
  - The deterministic regression is an engineering test of the tracked rules,
    not an independently human-approved factual benchmark.
  - The 100 formal Stardew cases remain machine_validated and require an
    independent reviewer before review_status may become approved.
  - Qwen/QLoRA training and GPU speculative-decoding benchmarks are supported
    by code and configs but were not executed in this offline build.
  - No model quality gain or speculative-decoding speedup is claimed.

See README.md, RELEASE_NOTES.md, RELEASE_VALIDATION.json, and docs/DELIVERY.md.
