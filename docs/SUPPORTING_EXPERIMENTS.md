# Supporting and Legacy Experiments

The repository retains earlier course exercises:

- character-level TinyGPT training on Tiny Shakespeare;
- GPT-2 and OPT prefill/decode benchmarking;
- request-scheduling simulation;
- plotting and generic benchmark utilities.

These components demonstrate implementation history and provide sanity
baselines, but they are not the final small-model contribution. A
character-level model cannot directly draft for Qwen because its token IDs,
special tokens, chat template, context representation, and cache interface are
incompatible with the target.

The main from-scratch model is now `TinyQwenDraft`:

```text
exact Qwen target tokenizer
+ custom Qwen-like decoder
+ target-teacher sequence adaptation
+ persistent-cache speculative decoding
```

The final GameGuideLM research claim is defined by multi-game grounding, Qwen
model adaptation, pretrained/custom draft comparison, and speculative decoding.
The Shakespeare configuration remains available only as a small educational
smoke example and regression test.
