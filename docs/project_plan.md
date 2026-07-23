# Project Plan

## Current status

The repository now has two functioning, tested tracks rather than a collection of
placeholders:

1. **Autoregressive inference experiments**
   - Character-level TinyGPT model, configurable training, checkpointing, and text generation
   - Cached greedy decoding and deterministic speculative decoding
   - Exact-token prompt construction
   - Prefill/decode benchmark measurement with TTFT, TPOT, throughput, and peak CUDA memory
   - Resumable YAML-driven benchmark sweeps and plotting
   - Prefill/decode serving simulation with FCFS and shortest-output-first policies

2. **Terraria structured knowledge**
   - Cleaned Item, NPC, Recipe, and Drop snapshots
   - Recipe-to-Item and Drop-to-entity linking
   - Referential-integrity audit
   - Compressed, indexed SQLite query database with FTS5
   - `TerrariaQueryStore` and `TerrariaFactService`
   - One-command rebuild and integration tests

## Phase 1 — Tiny GPT baseline

**Status: implemented.**

- Train TinyGPT on Tiny Shakespeare from `configs/tiny_gpt.yaml`
- Save model, tokenizer, and training report
- Generate text with temperature and top-k sampling
- Keep the implementation small enough for debugging and educational inspection

## Phase 2 — Prefill/decode measurement

**Status: implemented; real model measurements require local/Hugging Face checkpoints.**

- Generate exact token-length prompts
- Measure time to first token, mean time per output token, total latency, throughput,
  forward-call count, and peak accelerator memory
- Run one case or a resumable Cartesian sweep from YAML
- Export JSON/JSONL records and plot CSV/PNG summaries

## Phase 3 — Inference optimization

**Status: core implementations complete.**

- Cached greedy decoding
- Greedy speculative decoding with cache cropping and mismatch correction
- Phase-separated prefill/decode simulation
- FCFS and shortest-output-first scheduling policies

Next experiments:

- Continuous batching against a real inference runtime
- Paged KV-cache accounting
- Quantized draft/target comparisons
- CUDA profiler traces and kernel-level analysis

## Phase 4 — Terraria knowledge system

**Status: implemented and integration-tested.**

- Rebuild linked data and SQLite with `scripts/build_terraria_knowledge.py`
- Query facts with `scripts/query_terraria.py`
- Route natural-language intent into deterministic facts, then supply those facts to an LLM

Next work:

- Natural-language intent/entity parsing
- Answer rendering with evidence-aware prompts
- Evaluation set for factual exactness, ambiguity handling, and fallback behavior
- Optional RAG layer for long-form strategy guides that are not suitable for structured tables

## Phase 5 — Evaluation and report

**Status: framework ready; experimental runs remain.**

- Run the same prompt/output matrix across GPT-2, OPT, Qwen, and local fine-tuned models
- Report TTFT, TPOT, throughput, memory, and tail latency
- Compare baseline and speculative decoding under controlled tokenizer-compatible pairs
- Record hardware, software versions, seeds, model revisions, and raw result artifacts
