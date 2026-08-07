# Final Project: GameGuideLM

## Research objective

GameGuideLM studies how a language-model system can answer game-guide questions
when factual knowledge is external, versioned, and game-specific. Terraria and
Stardew Valley expose structured facts and document evidence through one common
contract. A fixed Qwen3-4B target generates the final answer, while compatible
small models are studied as speculative drafts.

The project is intentionally model-centric. Data pipelines construct trustworthy
evidence and evaluation workloads; the main experiments concern grounded
generation, evidence-aware target adaptation, draft/target alignment, and
speculative decoding.

## Contributions

1. **Game-agnostic evidence contract.** Different game schemas produce the same
   model-facing status, intent, facts, warnings, conditions, and provenance.
2. **Shared target/draft runtime.** Target-only, draft-only, and speculative
   generation use one loader, chat runtime, validation layer, and decoder.
3. **Evidence-aware training data.** Training prompts contain the exact bounded
   evidence package used at inference. Models learn evidence use and refusal,
   not Wiki memorization.
4. **Reliable pretrained draft baseline.** Qwen3-0.6B is compared with the fixed
   Qwen3-4B target under an exact tokenizer contract.
5. **Custom draft implemented from scratch.** `TinyQwenDraft` is a compact
   Qwen-token-compatible PyTorch decoder with tied embeddings, RMSNorm,
   grouped-query attention, RoPE, SwiGLU, and persistent KV cache.
6. **Persistent-cache speculative decoder.** Both models prefill once; mismatch
   correction and accepted bonus tokens keep their caches synchronized while
   preserving exact greedy target output.
7. **Model-pair analysis.** The project measures token agreement, top-k overlap,
   entropy, JS divergence, target-token likelihood, acceptance, and latency.
8. **Reliable multi-game workloads.** Terraria supplies broad structured
   coverage; Stardew adds seasonal, temporal, weather, route, bundle-mode, and
   player-state conditions.
9. **Safety fallback.** Invalid citations, unsupported URLs, failed generation,
   or missing evidence return the deterministic evidence answer.

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
Evidence-conditioned target prompt
           ↓
Qwen3-4B target-only
or compatible draft -> Qwen3-4B speculative
           ↓
Citation and unsupported-claim validation
           ↓
Valid LLM answer or deterministic fallback
```

## Model tracks

### Target

```text
Qwen3-4B
```

The target determines final answer quality and the exact greedy output.
Evidence-aware LoRA is optional and must be evaluated with retrieval held fixed.

### Pretrained draft baseline

```text
Qwen3-0.6B -> Qwen3-4B
```

This is the reliable speculative baseline. An optional teacher-answer LoRA can
adapt the 0.6B model to the fixed target's grounded continuation style.

### Custom from-scratch draft

```text
TinyQwenDraft -> Qwen3-4B
```

The custom model uses the exact target tokenizer and is trained on validated
target-generated continuations. It exists to test whether much lower draft cost
can compensate for lower token agreement. It is not assumed to be faster.

The legacy character-level Shakespeare TinyGPT remains an educational artifact,
not the main small-model contribution.

## Training pipeline

```text
Reviewed training annotations
           ↓
Run real game plug-in retrieval
           ↓
Store exact evidence-conditioned prompt
           ↓
Generate or verify cited target continuation
           ↓
Target LoRA / pretrained-draft LoRA / custom-draft training
```

Three model-training experiments are supported:

- **Target LoRA:** teach Qwen3-4B citation adherence, refusal, and concise
  organization.
- **Pretrained draft adaptation:** teach Qwen3-0.6B to reproduce the fixed
  target's grounded answer sequence.
- **Custom draft adaptation:** train `TinyQwenDraft` from random initialization
  on the same fixed-target sequences.

All current draft training is sequence-level assistant-token cross-entropy. It
is not claimed to be full-logit distillation.

## Why no new MoE

Adding MoE to an existing dense checkpoint would require architecture changes,
expert initialization, routing supervision, and substantially more training.
GameGuideLM already has explicit game evidence routing. An untrained MoE would
make the project less correct and less interpretable, so it remains outside the
final scope.

## Evaluation matrix

| Dimension | Variants |
|---|---|
| Game | Terraria, Stardew Valley |
| Question type | fact, conditional fact, guide/progression, false premise, missing context |
| Language | English, Chinese |
| Generator | deterministic, ungrounded target, grounded target, target LoRA |
| Decoder | target autoregressive, draft standalone, speculative |
| Draft | Qwen3-0.6B, adapted Qwen3-0.6B, TinyQwenDraft |
| Draft length | 2, 4, 6, 8 |
| Prompt length | 256, 512, 1024, 2048 token buckets |

Primary metrics:

- expected-status and intent accuracy;
- required-fact coverage;
- forbidden-error rate;
- citation validity and refusal correctness;
- draft/target prefill, TTFT, TPOT, latency, tokens/s, and peak memory;
- top-1 agreement, top-k overlap, entropy, JS divergence, and target-token
  likelihood;
- proposed/accepted tokens, accepted tokens per round, and forward calls;
- exact-token equality with target-only greedy decoding.

## Reliability and claim boundary

Terraria has a large tracked structured snapshot and a configurable Official
Wiki guide corpus. Stardew Valley ships with a versioned source-linked course-release
snapshot and a separate guide pipeline; the teammate-maintained module extends
the same contracts without changing model code.

Claims must distinguish:

- a tested software path;
- a defined course-release data scope;
- an online corpus that must be built locally;
- a training configuration versus a trained checkpoint;
- exact decoding correctness versus measured speed;
- acceptance rate versus end-to-end latency.

The current implementation has offline tests for the architecture, cache,
loader, trainer, and decoder. Custom-draft quality and speed remain experimental
until GPU training and warm benchmarking are completed.


## Stardew course-release state

The tracked Stardew snapshot now contains 505 structured records and 317 acquisition relations. The offline guide seed contains 25 pages and produces 100 chunks. The deterministic regression suite contains 100 examples with a 50/50 English-Chinese split and controlled `found`, `needs_context`, `partial`, and `not_found` behavior. The current deterministic implementation passes all 100 regression cases and the repository passes 171 offline tests.

These are engineering regression results. The 100 benchmark records remain `machine_validated` with `human_review_required=true`; no reviewer identity or approval has been fabricated. Qwen/QLoRA quality gains and speculative-decoding speedups remain GPU experiments until checkpoints are trained and warm benchmarks are executed.
