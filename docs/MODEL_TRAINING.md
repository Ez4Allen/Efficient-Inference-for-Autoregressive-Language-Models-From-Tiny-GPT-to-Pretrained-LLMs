# Model Training Plan

## Baseline first

Do not train before recording these baselines:

1. deterministic evidence renderer;
2. base Qwen3-4B grounded generation;
3. base Qwen3-0.6B standalone generation;
4. training-free speculative decoding;
5. draft/target alignment metrics.

Without these results, LoRA gains cannot be attributed correctly.

## Evidence-aware target LoRA

Use `scripts/build_grounded_sft.py` to build examples whose user message contains the evidence package. Train with:

```bash
python -m src.training.train_sft \
  --config configs/gameguidelm_qwen3_4b_lora.yaml
```

Target objectives:

- cite valid evidence IDs;
- preserve not-found and needs-context decisions;
- summarize multiple chunks instead of copying them;
- avoid unsupported names, numbers, schedules, and mechanics;
- produce concise bilingual answers.

Do not use LoRA to memorize the Wiki.

## Draft sequence adaptation

Freeze the final target configuration first. Then generate validated teacher answers:

```bash
python scripts/generate_teacher_answers.py \
  --input <reviewed-jsonl> \
  --output data/gameguide/target_teacher_train.jsonl
```

Train the 0.6B model with:

```bash
python -m src.training.train_sft \
  --config configs/gameguidelm_qwen3_0_6b_teacher_lora.yaml
```

Evaluate the adapter by token-level agreement and speculative latency, not only by answer semantics or training loss.

## Future logits distillation

True distribution distillation would add a loss such as:

```text
L = CE(student, target tokens) + λ · KL(target logits || student logits)
```

The existing `train_sft.py` does not implement this. A future `distill_draft.py` should be a separate experiment so that sequence-level SFT is not mislabeled as logits distillation.
