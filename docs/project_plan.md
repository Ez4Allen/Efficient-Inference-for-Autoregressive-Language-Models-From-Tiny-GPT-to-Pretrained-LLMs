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
   - Grounded `TerrariaAssistant` with bilingual routing, clarification, context, and CLI
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

## Phase 4 — Terraria knowledge and grounded assistant

**Status: structured QA MVP implemented and integration-tested.**

- Rebuild linked data and SQLite with `scripts/build_terraria_knowledge.py`
- Query facts with `scripts/query_terraria.py`
- Route English and Chinese natural-language questions into deterministic facts
- Resolve common aliases and preserve same-name ambiguity
- Render grounded answers with warnings, provenance, and LLM-ready context
- Use the assistant through `scripts/chat_terraria.py`

Next work:

- Add a document retriever for progression and mechanics questions that are not represented by structured tables
- Evaluate route accuracy, retrieval recall, factual exactness, hallucination rate, and latency
- Train an optional QLoRA answer model to follow retrieved evidence rather than memorize the catalog
- Compare standard and speculative decoding on the grounded assistant workload

## Phase 5 — Evaluation and report

**Status: framework ready; experimental runs remain.**

- Run the same prompt/output matrix across GPT-2, OPT, Qwen, and local fine-tuned models
- Report TTFT, TPOT, throughput, memory, and tail latency
- Compare baseline and speculative decoding under controlled tokenizer-compatible pairs
- Record hardware, software versions, seeds, model revisions, and raw result artifacts

## Milestone: Guide corpus and hybrid retrieval

- [x] Official Wiki source manifest and attribution
- [x] Guide-category and core-mechanics page discovery
- [x] Rate-limited incremental MediaWiki import
- [x] Section-aware HTML cleaning
- [x] Paragraph-aware retrieval chunking
- [x] Static quality audit and diagnostic bundle
- [x] SQLite FTS5 guide index
- [x] English/Chinese guide query expansion
- [x] Assistant routing for progression, strategy, and mechanics
- [ ] First live-crawl diagnostic review
- [ ] Retrieval benchmark on held-out guide questions
- [ ] Optional embedding/hybrid reranker comparison
