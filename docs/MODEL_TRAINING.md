# Model Training Plan

## Baselines first

Do not train adapters or the custom draft before recording these baselines:

1. deterministic evidence renderer;
2. base Qwen3-4B grounded generation;
3. base Qwen3-4B ungrounded generation;
4. base Qwen3-0.6B standalone generation;
5. Qwen3-0.6B -> Qwen3-4B speculative decoding;
6. draft/target token-alignment metrics.

Without these measurements, later changes cannot be attributed to grounding,
model adaptation, or decoder behavior.

## Data separation

Use reviewed training annotations to generate training examples. Use held-out
validation annotations only for model selection. Never use the formal Terraria
or Stardew evaluation files for target LoRA, pretrained-draft LoRA, or custom
draft training.

Required separation:

```text
reviewed train annotations
    -> grounded SFT / target-teacher train

held-out validation annotations
    -> grounded SFT / target-teacher validation

formal evaluation annotations
    -> final quality and runtime measurement only
```

The target-teacher generator also records the requested split so train and
validation examples cannot silently receive the same label.

## Evidence-aware target LoRA

Use `scripts/build_grounded_sft.py` to build examples whose prompt contains the
same bounded evidence package used at inference time. Train with:

```bash
python -m src.training.train_sft \
  --config configs/gameguidelm_qwen3_4b_lora.yaml
```

Target objectives:

- cite valid evidence IDs;
- preserve not-found and needs-context decisions;
- synthesize multiple chunks instead of copying them blindly;
- avoid unsupported names, numbers, schedules, and mechanics;
- produce concise English and Chinese answers.

Do not use LoRA to memorize the Wiki. Mutable facts remain in the retrieval
layer.

## Target-teacher data

Freeze the exact target checkpoint, tokenizer, chat template, grounding prompt,
and evidence budget before generating teacher data. Then create separate train
and validation files:

```bash
python scripts/generate_teacher_answers.py \
  --input <reviewed-training-jsonl> \
  --output data/gameguide/target_teacher_train.jsonl \
  --split train \
  --config configs/gameguidelm_qwen3_pair.yaml

python scripts/generate_teacher_answers.py \
  --input <reviewed-validation-jsonl> \
  --output data/gameguide/target_teacher_validation.jsonl \
  --split validation \
  --config configs/gameguidelm_qwen3_pair.yaml
```

Only target answers that pass the grounded generation path are retained. A
change to the target checkpoint, tokenizer, chat template, or prompt policy
requires regeneration of the teacher data.

## Pretrained draft sequence adaptation

Train an optional LoRA on Qwen3-0.6B:

```bash
python -m src.training.train_sft \
  --config configs/gameguidelm_qwen3_0_6b_teacher_lora.yaml
```

Evaluate it by:

- next-token agreement with the fixed target;
- target-token log probability;
- speculative acceptance;
- accepted tokens per round;
- end-to-end latency.

Lower SFT loss alone is not evidence of a better speculative draft.

## Custom TinyQwenDraft training

`TinyQwenDraft` is initialized from random weights but uses the exact target
tokenizer and chat template. Train it on the same validated target-generated
continuations:

```bash
python scripts/train_tiny_qwen_draft.py \
  --config configs/tiny_qwen_draft.yaml
```

The default model is intentionally compact:

```text
hidden size:        256
layers:               6
attention heads:      4
KV heads:             2
intermediate size:  768
context limit:      4096
```

Its actual parameter count is computed after loading the tokenizer. The large
shared embedding/output matrix dominates the parameter count, so weight tying
is mandatory.

Training properties:

- prompt tokens are masked with `-100`;
- only target-generated assistant tokens contribute to loss;
- `loss_only=True` projects only supervised positions into vocabulary logits;
- the tokenizer mapping and chat template are fingerprinted in the checkpoint;
- the final checkpoint includes config, weights, and tokenizer files.

The final checkpoint path for the default config is:

```text
results/tiny_qwen_draft/final
```

Use it with:

```text
configs/gameguidelm_tiny_qwen_pair.yaml
```

## Why this is sequence adaptation, not logits distillation

Both current draft-training paths use target-generated token sequences and
assistant-token cross-entropy. They do not minimize a KL divergence against the
full target distribution.

True logits distillation would add a loss such as:

```text
L = CE(student, target tokens) + lambda * KL(target logits || student logits)
```

That requires a separate memory-aware teacher/student trainer and should not be
claimed as part of the current implementation.

## Required post-training checks

For each trained draft:

1. validate the exact tokenizer contract;
2. verify target-only/speculative greedy output equality;
3. run token-alignment analysis on held-out prompts;
4. benchmark draft lengths 2/4/6/8;
5. benchmark prompt-length buckets;
6. compare with the unadapted Qwen3-0.6B baseline;
7. report slowdowns as well as speedups;
8. record model size, dtype, GPU, seed, checkpoint hash, and dataset hashes.
