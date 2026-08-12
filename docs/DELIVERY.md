# GameGuideLM v1.2.0 — Engineering Delivery

## Delivery status

```text
Offline source compilation:                 passed
Repository tests:                           206 passed
Release validator:                          passed
Stardew release validator:                  passed
Stardew deterministic regression:           100 / 100 passed
Stardew structured records:                 505
Stardew acquisition relations:              317
Stardew guide seed:                          25 pages / 100 chunks
Formal Stardew files:                        40 validation / 60 eval
Grounded Stardew training data:              159 train / 17 validation
Legacy Stardew SFT audit:                    1,262 pending candidates
Terraria resolved structured references:    14,353
Independent human benchmark approval:       pending
Professor-feedback GPU study:                code/config/notebook ready
```

The delivery is executable and testable offline. Expensive teacher generation,
pretraining, distillation, diversity sampling, and speculative acceptance are
explicit GPU stages and are not hidden inside tests or package import side
effects.

## What v1.2.0 adds

### Standard quality benchmarks

Frozen model answers can be compared with formal references using ROUGE-L,
chrF, token F1, and optional multilingual BERTScore. When deterministic rows are
provided, the same held-out IDs also receive a deterministic evidence-renderer
baseline. These metrics are reported beside the project-specific pass rate,
required-fact coverage, forbidden-error, citation, and unsupported-number checks.

### Transparent answer validation

`docs/ANSWER_VALIDATION.md` and `scripts/explain_answer_validation.py` expose:

- each required fact and whether/how it matched;
- every forbidden-error check;
- citation requirement, presence, and validity;
- unsupported numeric claims;
- the exact conjunction used to assign pass/fail.

### Prompt and answer limits

`audit_prompt_answer_sizes.py` reports configured budgets and observed
min/median/p90/p95/p99/max prompt and answer token counts, rather than reporting
means alone.

### Controlled custom-model study

The team-built 43.5M architecture is evaluated with a fixed Qwen3-0.6B tokenizer
and teacher under three controlled paths:

```text
scratch_distill
lightweight causal pretraining -> distillation
pretraining -> distillation -> grounded game adaptation
```

The study reports alignment, standard reference quality, bilingual/task slices,
mode-collapse diagnostics, and exact speculative acceptance. Formal evaluation
prompts are stored only as `split=held_out` and never enter teacher generation or
student optimization.

## Main entry points

Offline release:

```bash
pip install -r requirements-dev.txt
python scripts/validate_release.py
```

Professor-feedback post-processing of frozen GameGuideLM answers:

```bash
python scripts/run_professor_feedback_evaluation.py \
  --quality-rows <final-quality-rows.jsonl-or-csv> \
  --references data/stardew/evaluation/stardew_eval_v1.jsonl \
  --output-dir results/professor_feedback
```

Custom-model GPU study:

```bash
pip install -r requirements-metrics.txt
python scripts/run_custom_model_study.py \
  --config configs/custom_model_study.yaml \
  --stage all
```

Colab: `notebooks/05_custom_model_study.ipynb`.

## Distributed artifacts

Included:

- source code, configs, tests, validators, and documentation;
- versioned Stardew/Terraria snapshots and offline guide seeds;
- self-contained Stardew HTML demonstration;
- custom-model corpus/prompt builders and controlled-study orchestration;
- report-ready table/figure renderers.

Not included:

- Qwen model weights;
- generated teacher continuations;
- trained TinyQwenStudent checkpoints;
- generated SQLite databases and runtime logs;
- claims of unexecuted quality gains or speedups.

## Claim boundary

The deterministic 100/100 result validates tracked engineering rules, not all
possible game questions. The formal Stardew records remain machine-validated.
Any model result must be tied to its frozen Git commit, hardware, dependency
versions, seed, prompt limits, answer limits, and exact evaluation split.
