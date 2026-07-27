# Optimization Notes

This snapshot was reviewed and refactored as an engineering pass over the uploaded
repository.

## Major changes

- Removed fixed `/content/llm_project` runtime paths and introduced project-root discovery.
- Replaced all `NotImplementedError` command placeholders with working CLIs.
- Added exact-token synthetic prompt generation for controlled benchmarks.
- Added portable GPU/device monitoring and repeated prefill/decode measurement.
- Added resumable benchmark sweeps, single-case runs, and benchmark/simulation plotting.
- Added configurable TinyGPT training and generation with validation, checkpoint reports,
  top-k sampling, gradient clipping, and optimized scaled dot-product attention.
- Implemented the prefill/decode scheduling simulator and policies.
- Consolidated Terraria rebuild logic into a reusable pipeline and thin CLI.
- Compressed large JSON payloads in SQLite, reducing the generated database from roughly
  90 MB to roughly 39 MB while keeping transparent query compatibility.
- Improved item/NPC ambiguity handling, fact-service warnings, FTS safety, and JSON decoding.
- Added a grounded Terraria Assistant with bilingual intent routing, alias resolution,
  ambiguity clarification, structured retrieval, evidence context, deterministic rendering,
  a pluggable answer generator, and an interactive CLI.
- Removed the obsolete hand-maintained Terraria structured-fact stack.
- Added packaging metadata, dependency groups, portable documentation, and broad unit and
  integration coverage.

## Validation performed in this environment

- Python bytecode compilation for `src`, `scripts`, and `tests`
- Full `pytest` suite using deterministic toy models, the tracked Terraria snapshot, and
  end-to-end natural-language Assistant queries
- Strict Terraria link/audit/SQLite rebuild
- Serving-simulator CLI smoke run

GPU checkpoint benchmarks and QLoRA training were not executed here because they require
external model checkpoints and a suitable CUDA environment. Their command paths and input
validation were implemented and compile-tested.

## 0.5.0 — Local Terraria guide retrieval

- Added a rate-limited Official Terraria Wiki MediaWiki importer.
- Added category-driven page discovery, explicit core mechanics pages, revision
  caching, and resumable raw snapshots.
- Added section-aware HTML cleaning, paragraph-aware chunking, static quality
  auditing, and SQLite FTS5 indexing.
- Added deterministic English/Chinese query expansion and guide retrieval.
- Integrated progression/mechanics routing into `TerrariaAssistant` while
  retaining `TerrariaFactService` for structured Item/NPC/Recipe/Drop queries.
- Added source URLs, revision IDs, quality flags, license metadata, and
  evidence-preserving extractive rendering.
- Added a diagnostic bundle command for post-crawl cleaning review.
## 1.0.0 — GameGuideLM final course-project architecture

- Added a game-agnostic evidence contract and plug-in interface.
- Reused the complete Terraria knowledge and guide stack as the reference game.
- Added a Stardew Valley plug-in with temporal, seasonal, weather, location,
  route, and player-state conditions.
- Unified deterministic and Qwen-backed answering across games.
- Added evidence-aware LoRA data generation, optional draft teacher adaptation,
  multi-game evaluation, and token-level draft/target alignment analysis.
- Repositioned TinyGPT, GPT-2, serving simulation, and generic benchmark code as
  supporting course experiments rather than the final project claim.
- Finalized reproducibility, licensing, metadata, and release validation.
