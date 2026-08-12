# Professor and Supervisor Feedback Response

This checklist maps every presentation question to an implemented code path,
report artifact, and claim boundary.

## 1. Additional benchmark beyond pass rate and required-fact coverage

Implemented:

```text
src/evaluation/reference_metrics.py
scripts/evaluate_reference_metrics.py
scripts/run_professor_feedback_evaluation.py
requirements-metrics.txt
```

Frozen predictions are additionally evaluated with ROUGE-L F1, chrF, token F1,
and optional multilingual BERTScore. The same table includes a matched
`deterministic_evidence_renderer` baseline on the IDs present in the frozen Qwen
quality run. These standard/reference baselines are reported next to—not instead
of—status, required-fact, forbidden-error, citation, and numeric-support checks.

## 2. Explain answer validation

Implemented:

```text
docs/ANSWER_VALIDATION.md
src/evaluation/gameguide_eval.py::build_answer_validation_trace
scripts/explain_answer_validation.py
```

The report separates two layers:

1. online grounding validation (citations, source IDs, URLs, supported numbers,
   answer length, and hidden reasoning markers);
2. offline benchmark scoring (status, intent, required facts, forbidden errors,
   citation validity, and unsupported numbers).

The benchmark pass formula and a worked per-fact trace are machine-readable.

## 3. Maximum prompt and answer sizes

Implemented:

```text
src/evaluation/size_audit.py
scripts/audit_prompt_answer_sizes.py
scripts/run_professor_feedback_evaluation.py
```

The report records both configured budgets and observed min/median/p90/p95/p99/
maximum values.  The final grounded quality configuration is bounded by source
count, evidence characters, answer characters, and generated tokens.
`TinyQwenStudent` separately reports architectural context length, training
sequence length, and held-out prompt buckets.

## 4. Clarify the pipeline

`docs/ARCHITECTURE.md` now separates:

### Online grounded inference

```text
question/state -> game plugin -> fact/guide retrieval -> evidence selection
-> Qwen3-4B generation -> runtime validator -> answer/fallback
```

### Offline custom-model experiment

```text
train-only corpus -> causal pretraining -> Qwen3-0.6B distillation
-> grounded adaptation -> held-out alignment/diversity evaluation
```

Retrieval determines **what information is allowed**.  Speculative decoding
changes only **how target tokens are computed**.

## 5. Demonstrate diversity/generalization of the team-built model

Implemented:

```text
src/evaluation/diversity_metrics.py
src/evaluation/model_pair_alignment.py
scripts/build_student_prompt_pool.py
scripts/evaluate_custom_model_study.py
scripts/run_custom_model_study.py
configs/custom_model_study.yaml
```

Three fixed-architecture 43.5M variants isolate pretraining and game adaptation.
The study uses formal evaluation prompts only under `split=held_out`, includes
English and Chinese slices, and reports:

- validation loss/perplexity;
- top-1/top-k teacher agreement;
- Jensen--Shannon divergence and entropy gap;
- teacher-token probability/NLL;
- exact speculative acceptance;
- ROUGE-L/chrF/token F1 against formal references;
- language/game/category/prompt-length slices;
- unique-output rate, Distinct-1/2, Self-BLEU, and repetition.

Diversity metrics are identified as mode-collapse diagnostics; creative
variation is not the primary objective of a teacher-aligned student.

## One-command post-processing

After final predictions are frozen:

```bash
python scripts/run_professor_feedback_evaluation.py \
  --quality-rows results/final_quality_rows.csv \
  --references data/stardew/evaluation/stardew_eval_v1.jsonl \
  --output-dir results/professor_feedback \
  --deterministic-rows results/evaluation/rows.jsonl \
  --trace-id stardew_eval_001
```

Add `--bertscore` after installing `requirements-metrics.txt`.
