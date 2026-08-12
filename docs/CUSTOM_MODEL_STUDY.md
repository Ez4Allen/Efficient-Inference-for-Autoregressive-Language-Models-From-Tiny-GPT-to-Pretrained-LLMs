# Custom TinyQwenStudent Study

## Goal

The original `TinyQwenDraft` result answered a narrow serving question: a
43.5M-parameter model could be implemented, trained, loaded, and used in exact
speculative decoding, but its acceptance against Qwen3-4B was only 7.86%.  The
revised study answers the broader course feedback: **what does the model built
by the team learn, does pretraining help, does it generalize, and does it
collapse to repetitive outputs?**

The same architecture is renamed `TinyQwenStudent` only for this controlled
study.  The implementation remains in `src/models/tiny_qwen_draft/` so old
checkpoints and serving code stay compatible.

## Fixed model contract

All variants use the same architecture and Qwen3-0.6B tokenizer:

```text
approximately 43.5M parameters
6 decoder layers
hidden size 256
4 attention heads / 2 KV heads
RMSNorm + RoPE + GQA + SwiGLU
tied input/output embeddings
persistent crop-able KV cache
```

Only the training sequence changes.

## Controlled variants

| Variant | Initialization | Qwen3-0.6B sequence distillation | Grounded game adaptation |
|---|---|---:|---:|
| `scratch_distill` | random | yes | no |
| `pretrain_distill` | project-local causal pretraining | yes | no |
| `game_adapted` | `pretrain_distill` | inherited | yes |

This supports two controlled comparisons:

1. `scratch_distill` versus `pretrain_distill`: effect of lightweight
   pretraining under the same teacher and architecture;
2. `pretrain_distill` versus `game_adapted`: effect of grounded domain
   adaptation.

## Data and leakage contract

### Lightweight project-local pretraining

`build_tiny_student_corpus.py` combines only tracked project resources:

- Terraria and Stardew training conversations;
- curated guide pages;
- structured catalog records;
- deterministic Chinese alias-bridge documents from tracked Stardew aliases.

The current repository produces about 5.7M approximate tokens before GPU
packing.  It is correctly described as **lightweight project-local causal
pretraining**, not foundation-model pretraining.

Formal evaluation questions are never inserted.  Catalog entities and aliases
matching explicit held-out entity metadata are removed.  The manifest records
all rejections and language/domain/source distributions.

### Distillation prompt pool

`build_student_prompt_pool.py` creates:

- `train` and `validation` prompts from non-evaluation QA files;
- deterministic Chinese prompts from tracked Stardew aliases, with held-out
  entities excluded;
- formal evaluation prompts under a separate `held_out` split.

Teacher-data generation selects only `train` or `validation`.  The `held_out`
split is evaluation-only and keeps the formal reference answer for ROUGE-L,
chrF, and token-F1 scoring.

## Training pipeline

```text
tracked train-only corpus
        |
        v
causal next-token pretraining
        |
        +-------------------------------+
        |                               |
        v                               v
Qwen3-0.6B teacher continuations   random initialization
        |                               |
        +-------- sequence distillation-+
                         |
                         v
           grounded GameGuide adaptation
                         |
                         v
      held-out alignment/diversity evaluation
```

The distillation objective is assistant-token cross entropy on fixed greedy
Qwen3-0.6B continuations.  The grounded adaptation stage executes the real
retrieval pipeline and trains on validated evidence-conditioned teacher
answers.  Answer-preserving truncation keeps assistant tokens when a prompt is
long.

## Evaluation

Evaluation uses only the `held_out` split and reports:

### Teacher alignment

- top-1 agreement;
- top-k overlap;
- Jensen--Shannon divergence;
- student and teacher entropy plus entropy gap;
- probability and NLL/perplexity of teacher tokens under the student;
- exact speculative acceptance when the student drafts for Qwen3-0.6B.

### Standard reference benchmarks

- ROUGE-L F1;
- chrF;
- token F1;
- optional multilingual BERTScore through `requirements-metrics.txt`.

These complement rather than replace project-specific fact/citation checks.

### Diversity and generalization

- English/Chinese, game, category/intent, prompt-type, and prompt-length slices;
- conditional unique top-1 token ratio;
- unique sampled-output rate;
- Distinct-1 and Distinct-2;
- Self-BLEU;
- trigram repetition rate.

Sampled diversity is only a mode-collapse diagnostic.  A student/draft model is
supposed to align with the teacher, not maximize creative variation.

## Run

Edit `configs/custom_model_study.yaml` and run the resumable pipeline:

```bash
python scripts/run_custom_model_study.py \
  --config configs/custom_model_study.yaml \
  --stage all
```

Each stage can also be run independently:

```text
corpus
prompts
teacher
pretrain
scratch_distill
pretrain_distill
grounded_teacher
game_adapt
evaluate
render
```

GPU checkpoints and generated datasets remain under the ignored `results/`
directory (or an absolute Drive path configured by the user).  The repository
tracks code, configuration, tests, and manifests—not model weights.
