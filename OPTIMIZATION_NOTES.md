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
