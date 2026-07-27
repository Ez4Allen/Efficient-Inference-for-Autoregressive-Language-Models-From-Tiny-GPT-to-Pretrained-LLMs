# Project Notes

## Final direction

The final project is **GameGuideLM**, not a generic infrastructure benchmark.
The central artifact is a grounded multi-game Qwen language model with:

- Terraria and Stardew Valley knowledge plug-ins;
- evidence-conditioned generation and safe fallback;
- optional evidence-aware LoRA;
- Qwen3-0.6B/Qwen3-4B token-distribution analysis;
- correctness-first speculative decoding experiments.

## Supporting work

TinyGPT, GPT-2 timing, serving simulation, and generic plotting remain useful as
course-development history and supporting experiments, but they are not the
main final-project claim.

## Course connections

- Autoregressive language modeling;
- Transformer decoding;
- Retrieval-augmented and grounded generation;
- Supervised fine-tuning and LoRA/QLoRA;
- Knowledge distillation and draft/target alignment;
- Hallucination, factuality, and evaluation;
- Speculative decoding as a model-pair inference method.
