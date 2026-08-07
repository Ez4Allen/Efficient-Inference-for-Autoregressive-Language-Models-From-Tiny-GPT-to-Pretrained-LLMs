# Project Plan

## Current status

The repository now has three connected tracks:

1. **Grounded multi-game question answering**
   - Terraria and Stardew Valley plug-ins expose one evidence contract
   - Structured facts and Wiki guide retrieval remain external to model weights
   - Deterministic answers, citation validation, repair, and safe fallback

2. **Target-model training and evaluation**
   - Qwen3-4B grounded and ungrounded baselines
   - Evidence-aware SFT/QLoRA data construction
   - Multi-game quality evaluation and warm latency benchmarking

3. **Draft-model and speculative-decoding research**
   - Reliable pretrained baseline: Qwen3-0.6B -> Qwen3-4B
   - Custom from-scratch `TinyQwenDraft` -> Qwen3-4B
   - Exact tokenizer contracts, target-teacher sequence adaptation, token-level
     alignment analysis, and persistent-cache greedy speculative decoding

The older character-level Shakespeare TinyGPT remains a legacy educational
exercise. It is no longer the main small-model contribution.

## Phase 1 — Grounded workload

**Status: implemented; Stardew data expansion is being developed separately.**

- Build Terraria and Stardew structured stores
- Build game-specific Wiki guide indexes
- Route bilingual fact, conditional, progression, ambiguity, and false-premise
  questions
- Render evidence with stable source IDs and provenance
- Maintain reviewed training, validation, and evaluation splits without leakage

## Phase 2 — Target baseline and evidence-aware training

**Status: software implemented; GPU experiments remain.**

- Record deterministic and ungrounded baselines
- Run grounded Qwen3-4B generation
- Build evidence-conditioned SFT records from reviewed **training** annotations
- Train optional Qwen3-4B LoRA for citation following, refusal, and concise
  evidence synthesis
- Keep formal validation/evaluation files out of training

## Phase 3 — Reliable speculative baseline

**Status: implemented; warm GPU measurements remain.**

- Load Qwen3-0.6B and Qwen3-4B with the exact same tokenizer mapping
- Prefill draft and target once
- Reuse persistent KV caches across speculative rounds
- Crop rejected suffixes after mismatch
- Synchronize target correction and bonus tokens into both caches
- Preserve exact target-only greedy output
- Measure acceptance, target-call reduction, TTFT, TPOT, latency, throughput, and
  peak memory

No speedup is claimed until repeated warm measurements show that saved target
work exceeds draft overhead.

## Phase 4 — Custom TinyQwenDraft

**Status: architecture, loader, trainer, cache, and tests implemented; training
and benchmark results remain.**

- Use the fixed target tokenizer and record its exact vocabulary/chat-template
  fingerprints
- Train a compact Qwen-like decoder from random initialization
- Use tied embeddings, RMSNorm, grouped-query attention, RoPE, SwiGLU, and KV
  cache
- Supervise only target-generated assistant tokens
- Compare the custom draft with Qwen3-0.6B under identical prompts and target
- Keep Qwen3-0.6B as the reliable baseline even if the custom draft performs well

Required systems:

```text
A. Qwen3-4B target-only greedy
B. Qwen3-0.6B -> Qwen3-4B speculative
C. TinyQwenDraft -> Qwen3-4B speculative
```

## Phase 5 — Model-pair analysis

**Status: framework implemented; full experiment runs remain.**

- Top-1 token agreement
- Top-k overlap
- Draft and target entropy
- Jensen-Shannon divergence
- Target-token likelihood under each model
- Acceptance and accepted tokens per round
- Slices by game, language, question type, prompt length, and output token type

The primary custom-draft result is not training loss. It is whether its lower
cost produces a better end-to-end speed/acceptance trade-off than the pretrained
0.6B draft.

## Phase 6 — Final evaluation and report

**Status: protocol ready; results remain.**

- Grounded versus ungrounded target quality
- Base target versus evidence-aware target LoRA
- Pretrained versus custom draft
- Draft lengths 2/4/6/8
- Prompt lengths 256/512/1024/2048
- Exact target/speculative output equality
- Warm repeated latency with load time excluded
- Hardware, software, model/tokenizer revisions, seeds, and dataset hashes
- Manual review of citations, false premises, missing context, and unsupported
  claims

## Supporting experiments

The following remain useful implementation history but are not central final
claims:

- character-level Shakespeare TinyGPT;
- GPT-2/OPT prefill-decode benchmarks;
- request-scheduling simulation;
- plotting and generic benchmark utilities.
