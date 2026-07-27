# Final Project: GameGuideLM

## Research objective

GameGuideLM studies how a language model can provide reliable game-guide answers when factual knowledge is external, versioned, and game-specific. The project combines a shared Qwen model with game plug-ins that expose structured facts and document evidence through one contract.

The project is intentionally model-centric. Data pipelines exist to construct trustworthy evidence and evaluation workloads; the main experiments concern grounded generation, LoRA behavior, draft/target distribution alignment, and speculative decoding.

## Contributions

1. **A game-agnostic evidence contract.** Terraria and Stardew Valley use different schemas, but both produce the same model-facing fields: status, intent, facts, warnings, player-state conditions, and provenance.
2. **A shared Qwen runtime.** Qwen3-4B target and Qwen3-0.6B draft use one loader, one chat runtime, one validation layer, and one speculative implementation.
3. **Evidence-aware LoRA data construction.** Training examples contain the exact evidence package seen at inference time. The model learns evidence use and refusal, not factual memorization.
4. **Model-pair analysis.** The project measures top-1 agreement, top-k overlap, entropy, JS divergence, and target-token likelihood under each model.
5. **Reliable multi-game workloads.** Terraria is the full-scale implementation; Stardew Valley adds seasonal, temporal, weather, route, and player-state conditions that cannot be represented by Terraria-specific tables.
6. **Safety fallback.** Answers with invalid citations, unsupported URLs, missing evidence, or runtime errors return the deterministic evidence answer.

## End-to-end pipeline

```text
Question + game + player state
           ↓
Game plug-in router
           ↓
Structured FactService and/or guide FTS
           ↓
Standard evidence bundle with provenance
           ↓
Deterministic answer (safety baseline)
           ↓
Evidence-conditioned Qwen prompt
           ↓
Target / draft / speculative engine
           ↓
Citation validation
           ↓
Valid LLM answer or deterministic fallback
```

## Training pipeline

```text
Reviewed QA annotations
           ↓
Run game plug-in retrieval
           ↓
Store exact evidence-conditioned prompt
           ↓
Target answer with citations
           ↓
Existing QLoRA SFT trainer
```

Two model experiments are supported:

- **Target LoRA:** teach Qwen3-4B citation adherence, refusal, and concise organization.
- **Draft teacher LoRA:** teach Qwen3-0.6B to reproduce the fixed target's grounded answer sequences, then evaluate whether token agreement and speculative acceptance improve.

The current trainer performs standard assistant-token cross-entropy. It does not claim to implement logits distillation.

## Why no new MoE

Adding an MoE layer to an existing dense Qwen checkpoint is not a small fine-tuning change. It would require architecture changes, expert initialization, routing supervision, and substantially more training. GameGuideLM already has explicit game knowledge routing; a speculative, untrained MoE would make the project less correct and less interpretable. The final design therefore uses one shared model plus plug-in evidence and optional LoRA.

## Evaluation matrix

| Dimension | Variants |
|---|---|
| Game | Terraria, Stardew Valley |
| Question type | structured fact, conditional fact, guide/progression, false premise, missing context |
| Language | English, Chinese |
| Generator | deterministic, base target, target LoRA |
| Decoder | target autoregressive, draft standalone, speculative |
| Draft | base 0.6B, optional teacher-LoRA 0.6B |

Primary metrics:

- expected-status and intent accuracy;
- required-fact coverage;
- forbidden-error rate;
- citation validity and refusal correctness;
- TTFT, TPOT, latency, tokens/s, and peak memory;
- target/draft top-1 agreement and JS divergence;
- speculative acceptance rate and accepted tokens per round;
- exact-token equality with target greedy decoding.

## Reliability boundary

Terraria has a large tracked structured snapshot and a configurable Official Wiki guide corpus. Stardew Valley ships with a compact, source-linked starter snapshot and a separate guide pipeline. The Stardew plug-in is designed to accept the teammate-maintained full snapshot without changing the model code.

Claims should always distinguish:

- a tested code path;
- a compact included data scope;
- an online corpus that must be built locally;
- an optional GPU training experiment not yet run.
