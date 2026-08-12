# GameGuideLM

**A grounded multi-game language model for reliable game-guide question answering, evidence-aware LoRA training, and draft/target model analysis.**

GameGuideLM treats mutable game knowledge as evidence rather than asking a language model to memorize an entire Wiki. A game plug-in retrieves structured facts and guide excerpts, the target model turns that evidence into a natural answer, and a conservative validator rejects unsupported generations. The serving study compares Qwen3-0.6B and a custom `TinyQwenDraft` against Qwen3-4B; the professor-feedback extension evaluates the same 43.5M architecture as `TinyQwenStudent` under controlled Qwen3-0.6B pretraining/distillation ablations.

The final course-project emphasis is **LLM modeling and grounded generation**:

1. evidence-conditioned multi-game prompting;
2. one shared runtime for target-only, draft-only, and speculative generation;
3. optional multi-game LoRA for citation following and answer style;
4. target-teacher sequence adaptation for both pretrained and custom drafts;
5. a Qwen-token-compatible decoder-only draft model implemented in PyTorch;
6. token-level alignment analysis and end-to-end speculative-decoding experiments.

The databases are not the final model. They are the verifiable memory and evaluation workload used by the model.

---

## Supported games

### Terraria

Terraria is the full reference implementation:

- 6,283 cleaned Item records;
- 770 NPC records;
- 3,409 Recipe records with 4,221 variants;
- 3,144 Drop records;
- entity linking, provenance, integrity auditing, and SQLite FTS;
- an Official Wiki guide corpus for progression and mechanics questions;
- bilingual deterministic routing and safe refusal behavior.

### Stardew Valley

Stardew Valley is the completed second game plug-in and the primary demonstration workload:

- 505 versioned structured records: 41 crops, 55 fish, 34 villagers, 117 recipes, 30 Standard Bundles, and 228 acquisition entities;
- 317 structured acquisition relations;
- player-state-aware crop deadlines and fish availability;
- bilingual entity aliases, routing, answers, and safe refusals;
- 25 project-authored offline guide pages producing 100 searchable chunks;
- a 100-case bilingual deterministic regression suite with controlled intent/status distributions;
- 176 evidence-conditioned training records split 159/17 with the formal evaluation files excluded;
- a cleaned audit of 1,262 legacy AI-assisted SFT candidates, all honestly retained as `pending` and `verified=false`;
- a self-contained HTML showcase in `demo/stardew_showcase.html`.

This is a course-release snapshot, not a claim that every page or mechanic in the full Wiki is represented. Standard Bundle coverage is complete for the snapshot; Remixed Bundle requests are explicitly returned as `partial` rather than inferred. The benchmark passes engineering validation but remains marked for independent human source review before it can be called human-approved.

---

## Core model pipeline

```text
User question + selected game + optional player state
                         │
                         ▼
               Game-specific plug-in
         ┌───────────────┴────────────────┐
         │                                │
Structured FactService             Guide FTS retriever
(items, crops, fish,              (progression, strategy,
recipes, gifts, drops)             mechanics, walkthroughs)
         │                                │
         └───────────────┬────────────────┘
                         ▼
              Standard evidence bundle
        (facts, conditions, warnings, provenance)
                         ▼
        Prompt-budgeted evidence selection
       (stable source IDs, bounded context)
                         ▼
              Evidence-conditioned prompt
                         ▼
      Qwen3-4B target / compatible draft model
       (Qwen3-0.6B or custom TinyQwenDraft)
                         ▼
          Citation and unsupported-claim validator
                         ▼
      Valid answer / constrained repair / safe fallback
```

All games share the same model runtime. Game plug-ins own only knowledge, conditions, retrieval, and deterministic evidence rendering.

### Prompt budgeting and repair

Long Wiki evidence is selected under deterministic source and character budgets before it reaches Qwen. Citation IDs remain stable, guide chunks are trimmed at natural boundaries, and oversized structured objects use the verified deterministic summary rather than malformed truncated JSON. The grounding validator accepts only sources actually included in the prompt.

If the first model answer fails citation, URL, numeric, length, or thinking-trace checks, GameGuideLM performs one constrained rewrite with the same evidence. A second failure returns the deterministic answer. This makes model-generation failures measurable without exposing unsupported output.

---

## Model architecture

### Target model

```text
Qwen/Qwen3-4B
```

The target model is responsible for final answer quality. The first baseline runs the unmodified post-trained checkpoint. A multi-game evidence-aware LoRA is optional and should be trained only after the deterministic retrieval/evaluation pipeline is stable.

### Reliable draft baseline

```text
Qwen/Qwen3-0.6B
```

The pretrained draft remains the reliable baseline. The runtime loads the exact target tokenizer for both endpoints and verifies the complete token-to-ID mapping before speculative generation. It supports:

- independent small-model generation;
- training-free speculative decoding;
- optional sequence-level teacher LoRA using answers generated by the fixed 4B target;
- token-level agreement, entropy, top-k overlap, and JS-divergence analysis.

Speculative verification has two explicit modes. `exact` reuses the target's
one-token greedy path and guarantees target-token identity; `block` validates a
candidate block in one call for performance experiments and must report its
measured exact-match rate. See `docs/SPECULATIVE_CORRECTNESS.md`.

### Custom from-scratch draft

```text
TinyQwenDraft -> Qwen/Qwen3-4B
```

`src/models/tiny_qwen_draft/` contains a compact decoder-only language model written directly in PyTorch. It uses the target tokenizer, tied embeddings, RMSNorm, grouped-query attention, rotary position embeddings, SwiGLU, and a persistent crop-able KV cache. Its purpose is not to replace the target: it proposes tokens that the target verifies.

The custom model is trained on validated target-generated continuations, not on Shakespeare and not on the formal evaluation files. It is a research draft, so its usefulness must be established by acceptance rate and end-to-end latency rather than training loss. The Qwen3-0.6B baseline is retained even if the custom model is trained successfully.

See `docs/TINY_QWEN_DRAFT.md` for the architecture, tokenizer contract, training workflow, and benchmark protocol.

### Professor-feedback custom-model study

The same 43.5M-parameter architecture is now evaluated as `TinyQwenStudent`
against a fixed Qwen3-0.6B teacher.  The controlled study isolates three stages:

```text
scratch_distill
project-local causal pretraining -> distillation
pretraining -> distillation -> grounded game adaptation
```

The model architecture and tokenizer stay fixed.  Evaluation reports validation
perplexity, top-1/top-k agreement, Jensen-Shannon divergence, entropy gap,
teacher-token likelihood, exact speculative acceptance, ROUGE-L, chrF, token
F1, language/domain/category slices, and sampled-output diversity diagnostics.
This directly addresses generalization and mode-collapse questions without
misrepresenting creative diversity as the main objective of a draft/student
model.

Run the resumable study with:

```bash
python scripts/run_custom_model_study.py --stage all \
  --config configs/custom_model_study.yaml
```

See `docs/CUSTOM_MODEL_STUDY.md` and
`docs/PROFESSOR_FEEDBACK_RESPONSE.md`.

Build the decontaminated corpus/prompt manifests without a GPU:

```bash
python scripts/build_tiny_student_corpus.py \
  --output results/custom_model_study/data/pretraining_corpus.jsonl \
  --manifest results/custom_model_study/data/pretraining_manifest.json

python scripts/build_student_prompt_pool.py \
  --train-input data/stardew/sft/train.jsonl data/terraria/terraria_train_v1.jsonl \
  --validation-input data/stardew/sft/validation.jsonl data/terraria/terraria_validation_v1.jsonl \
  --eval-input data/stardew/evaluation/stardew_eval_v1.jsonl data/terraria/terraria_eval.jsonl \
  --output results/custom_model_study/data/prompt_pool.jsonl \
  --manifest results/custom_model_study/data/prompt_pool_manifest.json \
  --augment-stardew-zh
```

Formal evaluation records are written only with `split=held_out`; teacher-data
generation selects `train` or `validation`, so the same prompt pool cannot
silently leak evaluation questions into optimization.

### Why no MoE was added

A new MoE architecture would require pretraining or substantial supervised routing data and would confound the course project's main comparisons. The final design uses explicit game plug-ins for evidence routing and a shared Qwen model. This is simpler, measurable, and compatible with the existing checkpoints. MoE can remain a future extension rather than an unvalidated project feature.

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q
```

Current offline test result:

```text
206 passed
```

Run the complete offline release check (compilation, tests, and both
structured database builds) without downloading model weights or Wiki pages:

```bash
python scripts/validate_release.py
```

GPU LoRA experiments require:

```bash
pip install -r requirements-training.txt
```

---


### Build the complete Stardew release

```bash
python scripts/build_stardew_release.py
```

That one command regenerates the structured snapshot, cleans the legacy SFT pool, rebuilds both SQLite stores, runs the 100-case deterministic regression suite, validates release contracts, writes the demo outputs and HTML dashboard, and runs the full test suite.

Open the self-contained showcase locally:

```text
demo/stardew_showcase.html
```

## Build the knowledge backends

### Terraria structured database

```bash
python scripts/build_terraria_knowledge.py --quiet
```

### Terraria guide corpus

Online build:

```bash
python scripts/build_terraria_guides.py
```

Offline rebuild after a raw snapshot exists:

```bash
python scripts/build_terraria_guides.py --offline
```

### Stardew structured database

```bash
python scripts/build_stardew_knowledge.py --quiet
```

### Stardew guide corpus

Small online smoke build:

```bash
python scripts/build_stardew_guides.py --max-pages 3
```

Full configured build:

```bash
python scripts/build_stardew_guides.py
```

Offline rebuild:

```bash
python scripts/build_stardew_guides.py --offline
```

Generated raw Wiki pages, chunks, reports, and SQLite databases are excluded from Git. Source manifests, compact snapshots, evaluation data, and attribution are tracked.

---

## Use the multi-game model

### Deterministic evidence answer

```bash
python scripts/chat_gameguide.py \
  --game terraria \
  "How do I craft Night's Edge?"

python scripts/chat_gameguide.py \
  --game stardew \
  "阿比盖尔喜欢什么礼物？"

python scripts/chat_gameguide.py \
  --game stardew \
  --season spring \
  --day 24 \
  "Can I still plant Parsnip in time?"
```

### Qwen3-4B grounded generation

```bash
python scripts/chat_gameguide.py \
  --game terraria \
  --llm \
  --engine target \
  "进入困难模式后该做什么？"

python scripts/chat_gameguide.py \
  --game stardew \
  --llm \
  --engine target \
  "What gifts does Abigail love?"
```

### Ungrounded model ablation

Use the same Qwen checkpoint without retrieved evidence only as a controlled
hallucination baseline:

```bash
python scripts/chat_gameguide.py \
  --game terraria \
  --ungrounded \
  --engine target \
  "How do I craft a Void Slime King?"
```

This mode is not the deployed assistant and intentionally bypasses grounding
validation so its errors can be measured.

### Draft and speculative modes

```bash
python scripts/chat_gameguide.py \
  --game terraria \
  --llm \
  --engine draft \
  "进入困难模式后该做什么？"

python scripts/chat_gameguide.py \
  --game terraria \
  --llm \
  --engine speculative \
  "进入困难模式后该做什么？"
```

The speculative implementation is a greedy decoder with persistent draft and target KV caches. Both models prefill once, rejected suffixes are cropped after a mismatch, and correction or bonus tokens are synchronized into both caches. Use `verification_mode: exact` to reproduce a deterministic target's one-token greedy sequence; use `verification_mode: block` for one-call block verification and report both wall-clock speed and measured token-ID exact-match rate.

---

## Evidence-aware LoRA

The existing `src/training/train_sft.py` remains the only QLoRA training implementation.

### 1. Build multi-game grounded SFT examples

```bash
python scripts/build_grounded_sft.py \
  --input data/terraria/terraria_train_v1.jsonl \
  --default-game terraria \
  --output data/gameguide/grounded_train.jsonl
```

For mixed input files, add a `game` field to every annotation rather than relying on `--default-game`. Add the reviewed Stardew **training** split only after the Stardew data-cleanup PR is complete. Never use `stardew_validation_v1.jsonl` or `stardew_eval_v1.jsonl` as training input.

### 2. Train the 4B evidence-following adapter

```bash
python -m src.training.train_sft \
  --config configs/gameguidelm_qwen3_4b_lora.yaml
```

This experiment measures whether LoRA improves citation adherence, refusal behavior, and answer organization. It is not used to memorize Wiki facts.

### 3. Optional draft teacher adaptation

Generate validated target answers from reviewed **training** annotations:

```bash
python scripts/generate_teacher_answers.py \
  --input data/terraria/terraria_train_v1.jsonl \
  --output data/gameguide/target_teacher_train.jsonl \
  --split train
```

Create a separate teacher-validation file from held-out validation annotations:

```bash
python scripts/generate_teacher_answers.py \
  --input data/terraria/terraria_validation_v1.jsonl \
  --output data/gameguide/target_teacher_validation.jsonl \
  --split validation
```

Train the pretrained 0.6B draft adapter:

```bash
python -m src.training.train_sft \
  --config configs/gameguidelm_qwen3_0_6b_teacher_lora.yaml
```

Train the custom from-scratch draft on the same target-generated continuations:

```bash
python scripts/train_tiny_qwen_draft.py \
  --config configs/tiny_qwen_draft.yaml
```

This is sequence-level target adaptation. A future logits-distillation experiment would require a separate KL-based loss implementation and is not represented as part of either current trainer.

---

## Model-pair analysis

Analyze either compatible draft and Qwen3-4B on the exact same grounded completion. Use `configs/gameguidelm_qwen3_pair.yaml` for the pretrained baseline or `configs/gameguidelm_tiny_qwen_pair.yaml` after training the custom draft:

```bash
python scripts/analyze_qwen_pair.py \
  --game terraria \
  "进入困难模式后该做什么？"
```

Reported model metrics include:

- top-1 next-token agreement;
- mean top-k overlap;
- draft and target entropy;
- Jensen-Shannon divergence;
- target-token log probability under each model.

These metrics explain speculative acceptance behavior at the model-distribution level, rather than treating the models as black-box APIs. The analyzer prefills the long prompt with KV cache and materializes logits only for completion positions, avoiding multi-gigabyte full-prompt logit tensors for Qwen's large vocabulary.

### Warm target/draft/speculative study

Use one process, load the pair once, warm every engine, and repeat each case:

```bash
python scripts/benchmark_gameguidelm.py \
  --input \
    data/terraria/terraria_eval.jsonl \
    data/stardew/evaluation/stardew_eval_v1.jsonl \
  --output results/model_study/exact_runs.jsonl \
  --summary results/model_study/exact_summary.json \
  --engines target draft speculative \
  --warmup-runs 1 \
  --runs 5 \
  --max-new-tokens 128 \
  --verification-mode exact
```

Use `--verification-mode exact` for the correctness gate. Run a separate,
otherwise identical command with `--verification-mode block` for wall-clock
speed measurements. The study records generated token IDs and hashes, first
mismatch position, token agreement, grounding validation, TTFT, TPOT, latency,
tokens/s, forward calls, acceptance, target determinism, environment metadata,
and the evidence-budget configuration. Exact mode intentionally performs
one-token target verification, so it is not a speed path. If block mode diverges
on a mixed-precision GPU, `attn_implementation: eager` on the target endpoint is
available as a diagnostic rerun; changing the attention backend also changes
latency, so keep reporting both token agreement and speed. Checkpoint download
and load time are excluded from warm generation metrics. Run the study once with
`configs/gameguidelm_qwen3_pair.yaml` and once with
`configs/gameguidelm_tiny_qwen_pair.yaml`; do not compare drafts measured under
different prompts, output limits, dtypes, attention backends, or warm-up settings.

---

## Evaluation

Build the relevant guide corpora before evaluating guide/progression examples.
Without a local guide database, the assistant correctly returns `not_found` for
those examples and the aggregate score will be lower by design.

Deterministic multi-game evaluation:

```bash
python scripts/evaluate_gameguidelm.py \
  --input \
    data/terraria/terraria_eval.jsonl \
    data/stardew/evaluation/stardew_eval_v1.jsonl \
  --output results/gameguidelm/deterministic.jsonl \
  --summary results/gameguidelm/deterministic_summary.json
```

Qwen target evaluation:

```bash
python scripts/evaluate_gameguidelm.py \
  --llm \
  --engine target \
  --input data/stardew/evaluation/stardew_eval_v1.jsonl \
  --output results/gameguidelm/qwen_target.jsonl \
  --summary results/gameguidelm/qwen_target_summary.json
```

Ungrounded target ablation:

```bash
python scripts/evaluate_gameguidelm.py \
  --ungrounded \
  --engine target \
  --input data/stardew/evaluation/stardew_eval_v1.jsonl \
  --output results/gameguidelm/qwen_ungrounded.jsonl \
  --summary results/gameguidelm/qwen_ungrounded_summary.json
```


The evaluator records:

- routing and expected-status accuracy;
- required-fact coverage;
- forbidden-error rate;
- citation presence;
- retrieved evidence count;
- runtime debug metrics when an LLM engine is enabled.

Automated lexical coverage is a regression signal, not a claim of human-level factual correctness. Final model comparisons should include manual review of refusal, citation, and false-premise cases.

---

## Recommended final experiments

### Experiment A — Grounding ablation

```text
Qwen3-4B without evidence
vs
Qwen3-4B with GameGuideLM evidence
```

Measure factual coverage, false-premise hallucination, citation validity, and human preference.

### Experiment B — Target LoRA

```text
Base Qwen3-4B grounded
vs
Evidence-aware Qwen3-4B LoRA grounded
```

Measure evidence adherence and answer quality while keeping retrieval fixed.

### Experiment C — Draft/target model analysis

Compare token agreement, entropy, top-k overlap, and divergence by:

- game;
- fact versus guide question;
- English versus Chinese;
- prompt length;
- factual names/numbers versus ordinary connective text.

### Experiment D — Pretrained speculative baseline

```text
Qwen3-4B autoregressive
vs
Qwen3-0.6B -> Qwen3-4B speculative
```

Measure exact-token correctness, TTFT, TPOT, end-to-end latency, target calls, draft calls, acceptance rate, accepted tokens per round, and peak memory.

### Experiment E — Optional pretrained-draft adaptation

```text
Base Qwen3-0.6B draft
vs
Qwen3-0.6B teacher-answer LoRA draft
```

Use the same fixed Qwen3-4B target. The useful result is not lower training loss; it is improved token agreement, speculative acceptance, and end-to-end latency.

### Experiment F — Custom draft study

```text
Qwen3-4B autoregressive
vs
TinyQwenDraft -> Qwen3-4B speculative
vs
Qwen3-0.6B -> Qwen3-4B speculative
```

The custom draft is evaluated as a speed/acceptance trade-off: fewer parameters and lower draft cost may be offset by lower token agreement. Test multiple draft lengths and prompt-length buckets, and report slowdowns as well as speedups.

---

## Repository layout

```text
src/gameguide/               Game-agnostic evidence, prompting, validation, Qwen orchestration
src/games/terraria/          Adapter for the complete Terraria implementation
src/games/stardew/           Stardew facts, player-state logic, guides, and deterministic assistant
src/assistant/               Existing Terraria-specific assistant implementation
src/knowledge/               Terraria structured catalog and FactService
src/retrieval/               Shared MediaWiki import, cleaning, chunking, quality, and FTS infrastructure
src/models/                  Shared loader plus the custom TinyQwenDraft implementation
src/inference/               Autoregressive and persistent-cache speculative decoding
src/training/                SFT/QLoRA plus custom-draft sequence-adaptation training
src/evaluation/              Multi-game QA and draft/target model analysis
scripts/                     Build, chat, train-data, evaluate, and model-analysis entry points
data/terraria/               Terraria snapshots, manifests, and reviewed QA
data/stardew/                Stardew compact snapshot, guide manifest, and reviewed QA seed
data/gameguide/              Generated evidence-conditioned model-training data
```

The Shakespeare character-level TinyGPT, GPT-2 benchmark, and serving-simulation code remain as supporting coursework and implementation history. Shakespeare is no longer the project's main small-model experiment; the custom Qwen-token-compatible draft is. See `docs/SUPPORTING_EXPERIMENTS.md`.

---

## Final project statement

> **GameGuideLM is a grounded multi-game language-model system that separates mutable game knowledge from model parameters, trains models to follow retrieved evidence, implements a Qwen-token-compatible draft from scratch, and compares pretrained and custom drafts against the same Qwen3-4B target on realistic Terraria and Stardew workloads.**

---

## Release documentation

- `docs/FINAL_PROJECT.md`: research framing and contribution boundaries;
- `docs/ARCHITECTURE.md`: exact game plug-in, evidence, Qwen, and validation pipeline;
- `docs/EXPERIMENTS.md`: model, LoRA, and speculative-decoding experiment matrix;
- `docs/MODEL_STUDY.md`: prompt-budget, warm benchmark, alignment, and reporting protocol;
- `docs/MODEL_TRAINING.md`: evidence-aware target and draft training plan;
- `docs/TINY_QWEN_DRAFT.md`: custom draft architecture, tokenizer contract, training, and benchmarking;
- `docs/STARDEW_MODULE.md`: Stardew capability and extension contract;
- `docs/REPRODUCIBILITY.md`: offline, online-corpus, and GPU experiment protocol;
- `docs/DELIVERY.md`: exact v1.2.0 scope, validation state, and claim boundaries;
- `docs/PROFESSOR_FEEDBACK_RESPONSE.md`: feedback-to-code/report artifact map;
- `docs/CUSTOM_MODEL_STUDY.md`: controlled TinyQwenStudent ablation and leakage contract;
- `PROFESSOR_FEEDBACK_IMPLEMENTATION_SUMMARY.md`: concise delivery and validation summary;
- `RELEASE_NOTES.md`: v1.2.0 scope and reproducibility statement.
