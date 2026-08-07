# GameGuideLM training data

The model is trained on **evidence-conditioned conversations**, not raw Wiki
text. Use `scripts/build_grounded_sft.py` to convert reviewed Terraria and
Stardew **training** annotations into the chat JSONL format consumed by
`src/training/train_sft.py`.

Keep the data roles separate:

```text
grounded_train.jsonl                 target evidence-aware LoRA

target_teacher_train.jsonl           pretrained/custom draft training
target_teacher_validation.jsonl      draft model selection

formal game validation/eval files    final evaluation only
```

`scripts/generate_teacher_answers.py` runs the fixed Qwen3-4B target on the real
grounded prompt and retains validated target continuations. The same teacher
files can be used for:

- Qwen3-0.6B sequence-level LoRA adaptation;
- from-scratch `TinyQwenDraft` sequence adaptation.

These are target-sequence adaptation experiments, not full-logit distillation.
Do not create teacher training data from formal validation/evaluation records,
and do not commit generated model checkpoints to this directory.
