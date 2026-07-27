# GameGuideLM v1.0.0

GameGuideLM v1.0.0 is the course-project release that consolidates the earlier
TinyGPT, Terraria knowledge, Wiki retrieval, and Qwen inference work into one
model-centric research system.

## Main contribution

A shared Qwen3 language model answers questions for multiple games from
versioned external evidence. Terraria and Stardew Valley use different game
schemas but expose one model-facing evidence contract. The same grounded
prompts are then used for LoRA and draft/target model analysis.

## Included

- Multi-game plug-in API and common evidence schema;
- Complete Terraria structured-fact integration;
- Terraria Official Wiki guide retrieval;
- Stardew Valley conditional fact module and guide pipeline;
- Qwen3-4B target and Qwen3-0.6B draft through one loader/runtime;
- Deterministic fallback and citation/URL/numeric grounding validation;
- Evidence-conditioned SFT data construction;
- Optional 4B evidence-following LoRA and 0.6B teacher-answer LoRA configs;
- Draft/target top-1 agreement, top-k overlap, entropy, and JS-divergence tools;
- Multi-game deterministic/Qwen evaluation scripts;
- Prompt-budgeted evidence selection and constrained generation repair;
- Warm repeated target/draft/speculative benchmark with output hashes and exact-match checks;
- Deterministic prompt-budgeted evidence selection with stable citation IDs;
- One constrained repair pass before deterministic fallback;
- Cache-aware completion-logit analysis for long grounded prompts;
- Warm repeated target/draft/speculative benchmark with output hashes and exact-match checks;
- Autoregressive and correctness-first speculative decoding baselines;
- Offline unit and integration tests.

## Deliberately not claimed

- The included Stardew structured snapshot is not full-Wiki coverage;
- sequence-level teacher LoRA is not logits distillation;
- the current speculative decoder is a correctness baseline, not an optimized
  speed result;
- automated lexical evaluation is not a replacement for human factual review;
- MoE is not included because no valid expert-routing training experiment was
  available for this project scope.

## Verified release state

- Offline test suite: **140 passed**;
- Terraria structured snapshot rebuild and integrity audit: passed;
- Stardew structured snapshot rebuild and deterministic crop-deadline smoke query: passed;
- Multi-game CLI/import smoke checks: passed;
- Release metadata and claim boundaries: `RELEASE_MANIFEST.json`.

## Reproducible release checks

```bash
pip install -r requirements-dev.txt
python -m pytest -q
python scripts/build_terraria_knowledge.py --quiet
python scripts/build_stardew_knowledge.py --quiet
```

Online Wiki corpora and Qwen checkpoints are built/downloaded separately and
are not distributed in the repository.
