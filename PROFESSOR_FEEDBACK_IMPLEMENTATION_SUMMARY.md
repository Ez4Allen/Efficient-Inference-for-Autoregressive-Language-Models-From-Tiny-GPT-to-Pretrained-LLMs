# Professor-Feedback Engineering Implementation Summary

## Objective

This v1.2.0 extension turns the presentation feedback into reproducible code,
not just report prose. It preserves the deployed GameGuideLM pipeline and adds a
controlled study of the 43.5M-parameter decoder implemented by the team.

## Feedback-to-artifact map

| Feedback | Engineering response |
|---|---|
| Add another benchmark | Matched deterministic evidence-renderer baseline plus ROUGE-L, chrF, token F1, optional multilingual BERTScore |
| Explain answer validation | Per-fact validation trace, documented online/offline layers, exact pass formula |
| Report max prompt/answer size | Configured-limit plus min/median/p90/p95/p99/max audit |
| Clarify pipeline | Separate online grounded inference and offline model-study diagrams/docs |
| Demonstrate custom-model diversity/generalization | Controlled pretrain/distill/game-adapt ablation, bilingual/task/length slices, entropy/JS/top-k and sampled diversity |

## Custom-model design

The implementation remains backward-compatible as `TinyQwenDraft`, with the
study alias `TinyQwenStudent`. All variants use the same Qwen3-0.6B tokenizer,
architecture, parameter count, held-out prompts, and evaluation settings.

```text
Variant A: random -> Qwen3-0.6B sequence distillation
Variant B: lightweight project-local causal pretraining -> same distillation
Variant C: Variant B -> grounded GameGuide adaptation
```

This isolates the effects of pretraining and domain adaptation. The pretraining
corpus is reproducible from tracked project resources and contains approximately
5.7M tokens; it is not represented as foundation-model pretraining.

## Evaluation dimensions

- teacher top-1 and top-k agreement;
- Jensen--Shannon divergence, student/teacher entropy, and entropy gap;
- teacher-token probability, NLL/perplexity, and exact speculative acceptance;
- ROUGE-L, chrF, token F1, and optional BERTScore against formal references;
- English/Chinese, Stardew/Terraria, category, intent, prompt type, and prompt
  length slices;
- unique conditional top-1 ratio, unique sampled-output rate, Distinct-1/2,
  Self-BLEU, and trigram repetition.

Diversity is explicitly treated as a mode-collapse/generalization diagnostic,
not as a requirement for creative variation from a speculative student.

## Leakage controls

- formal evaluation rows are stored only with `split=held_out`;
- teacher continuation generation accepts only `train` or `validation`;
- exact evaluation questions are excluded from the optimization pool;
- catalog documents matching explicit held-out entities/aliases are excluded
  from project-local pretraining where metadata is available;
- deterministic Chinese augmentation uses only tracked aliases and excludes
  held-out Stardew entities;
- checkpoints and result data remain outside Git.

## Offline validation completed

```text
206 tests passed across six CI shards
source/script/test compilation passed
release validator passed
Stardew release validator passed
custom corpus smoke build passed
custom prompt-pool smoke build passed
```

Reproducible offline smoke counts:

```text
pretraining documents:      15,291
approximate tokens:          5,744,177
train / validation docs:     13,740 / 1,551
English / Chinese docs:      15,022 / 269
prompt-pool records:         1,690
train / validation / held:   1,416 / 184 / 90
English / Chinese prompts:   1,346 / 344
```

## GPU execution

Run `notebooks/05_custom_model_study.ipynb` or:

```bash
python scripts/run_custom_model_study.py   --config configs/custom_model_study.yaml   --stage all
```

The release intentionally does not claim the ablation result before this GPU
run completes. `render_custom_model_study_report.py` converts the frozen summary
into CSV, LaTeX, figures, and a report insert.
