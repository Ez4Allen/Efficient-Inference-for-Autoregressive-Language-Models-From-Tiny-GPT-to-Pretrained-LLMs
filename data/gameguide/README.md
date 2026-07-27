# GameGuideLM training data

The main model is trained on **evidence-conditioned conversations**, not on raw Wiki text. Use `scripts/build_grounded_sft.py` to convert reviewed Terraria and Stardew QA annotations into the chat JSONL format consumed by `src/training/train_sft.py`.

The optional `scripts/generate_teacher_answers.py` runs the fixed Qwen3-4B target on grounded prompts and creates sequence-level teacher data for the Qwen3-0.6B draft. This is an optional model-alignment experiment after the training-free speculative baseline is measured.
