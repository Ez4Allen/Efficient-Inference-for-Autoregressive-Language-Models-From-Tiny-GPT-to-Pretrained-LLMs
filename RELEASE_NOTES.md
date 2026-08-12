# GameGuideLM v1.2.0

GameGuideLM v1.2.0 is the professor-feedback engineering release. It preserves
all grounded multi-game QA and speculative-decoding work from v1.1.0 and adds a
controlled, reproducible study of the 43.5M-parameter model implemented by the
team. The release is designed to answer the questions raised after the
presentation: how answers are validated, what the actual prompt/answer limits
are, whether standard metrics support the project-specific score, how the
online/offline pipelines differ, and whether the custom model generalizes or
collapses.

## Professor-feedback deliverables

- Standard reference metrics beside required-fact coverage and pass rate:
  ROUGE-L F1, chrF, token F1, and optional multilingual BERTScore.
- A machine-readable answer-validation trace and a documented pass formula that
  separates online grounding validation from offline benchmark scoring.
- Configured and observed prompt/answer size audits with min, median, p90, p95,
  p99, maximum, and limit-violation counts.
- Separate online grounded-inference and offline custom-model-training
  pipelines in the architecture documentation.
- A controlled custom-model ablation with one fixed architecture and teacher:
  `scratch_distill`, `pretrain_distill`, and `game_adapted`.
- Generalization/mode-collapse diagnostics: English/Chinese, game, category,
  intent, prompt-type, and prompt-length slices; top-1/top-k agreement; entropy
  gap; Jensen--Shannon divergence; teacher-token likelihood; unique conditional
  top-1 ratio; Distinct-1/2; Self-BLEU; and repetition.
- Exact speculative acceptance against Qwen3-0.6B as an additional task-relevant
  measure for the team-built student.

## Custom-model training protocol

The fixed 43.5M-parameter Qwen-token-compatible architecture is evaluated under
three controlled training paths:

```text
A. random initialization -> Qwen3-0.6B sequence distillation
B. lightweight project-local causal pretraining -> the same distillation
C. path B -> grounded GameGuide adaptation
```

The project-local causal corpus contains approximately 5.7M tokens generated
only from tracked training conversations, structured catalogs, curated guide
text, and deterministic bilingual alias bridges. Formal evaluation questions
are kept under an evaluation-only `held_out` split. This stage is described as
**lightweight project-local causal pretraining**, not foundation-model
pretraining.

Run the resumable GPU study with:

```bash
python scripts/run_custom_model_study.py \
  --config configs/custom_model_study.yaml \
  --stage all
```

A Colab entry point is provided at `notebooks/05_custom_model_study.ipynb`.
Model weights, generated teacher data, and result artifacts remain outside Git.

## Core grounded-system release

The release continues to include:

- 505 versioned Stardew structured records and 317 acquisition relations;
- 25 offline guide pages producing 100 searchable chunks;
- 100 bilingual deterministic engineering-regression cases;
- 159/17 grounded training/validation records with formal evaluation isolated;
- 1,262 audited legacy SFT candidates, all retained as pending human review;
- the Terraria reference snapshot with 14,353 resolved structured references;
- Qwen3-4B target generation, Qwen3-0.6B draft baselines, and the custom
  TinyQwenDraft/TinyQwenStudent implementation;
- exact and block speculative-verification modes with persistent KV cache;
- one-command release validation and an offline interactive demonstration.

## Offline verification

The source release passed 206 tests across six CI shards:

```text
tests/assistant       34
tests/gameguide       38
tests/games           38
tests/knowledge        8
tests/retrieval        8
tests/test_*.py        80
                     ---
total                 206
```

It also passed:

```bash
python -m compileall -q src scripts tests
python scripts/validate_release.py --skip-pytest
python scripts/validate_stardew_release.py
```

The decontaminated corpus and prompt-pool builders are smoke-tested in CI.

## Claim boundaries

- The 100 Stardew cases are machine-validated engineering regressions and still
  require independent human source review before being called human-approved.
- Standard text-similarity metrics complement rather than replace fact,
  citation, and numeric-support checks.
- Sampled-output diversity is a mode-collapse diagnostic; the student model is
  optimized for teacher alignment, not creative variation.
- The tracked release contains code, configs, tests, and protocols. It does not
  include Qwen weights, generated teacher datasets, trained custom-model
  checkpoints, or fabricated GPU results.
- GPU conclusions must be reported from the frozen output directory together
  with commit hash, hardware, software versions, random seed, and exact prompt
  and generation settings.

## Reproduce the offline release

```bash
pip install -r requirements-dev.txt
python scripts/validate_release.py
```

For the custom-model GPU study:

```bash
pip install -r requirements-metrics.txt
python scripts/run_custom_model_study.py --stage all
```

## Historical v1.1.0 milestone

v1.1.0 completed the Stardew course-release workload and introduced the custom
Qwen-compatible draft, persistent-cache speculative decoding, exact tokenizer
contracts, and target-teacher sequence adaptation. v1.2.0 extends rather than
replaces those results.
