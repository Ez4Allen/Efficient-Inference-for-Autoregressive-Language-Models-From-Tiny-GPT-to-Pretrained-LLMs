# Changelog

## 1.1.0 — Stardew release and TinyQwenDraft foundation

- Expanded Stardew from a 31-record integration seed to 505 versioned structured records and 317 acquisition relations.
- Added all 30 Standard Bundles and explicit `partial` handling for unsupported Remixed Bundle coverage.
- Expanded the offline guide seed to 25 pages and 100 searchable chunks with bilingual query expansion.
- Added a 100-case bilingual deterministic regression suite with controlled category and status distributions; all 100 cases pass.
- Added 176 deterministic evidence-conditioned training records with formal evaluation isolation.
- Audited the 1,262 teammate SFT candidates, reset unverifiable review flags, and created source/template-disjoint development splits.
- Added one-command release build, release-contract validation, demo transcripts, and a self-contained HTML showcase.
- Expanded the full offline test suite from 158 to 171 tests.
- Preserved an explicit truth boundary: human benchmark review and GPU model/speed experiments remain pending.


- Added `TinyQwenDraft`, a Qwen-token-compatible decoder implemented directly
  in PyTorch with tied embeddings, RMSNorm, GQA, RoPE, SwiGLU, and KV cache.
- Added exact tokenizer vocabulary/chat-template fingerprinting and runtime
  contract validation.
- Added a from-scratch target-teacher training pipeline with answer-only loss and
  memory-aware supervised-position vocabulary projection.
- Reworked greedy speculative decoding to prefill draft and target once, retain
  persistent caches, crop rejected suffixes, and synchronize correction/bonus
  tokens.
- Added custom-draft runtime configuration, training configuration, smoke paths,
  and model/decoder tests.
- Reframed Shakespeare TinyGPT as a legacy educational example rather than the
  main small-model contribution.
- Expanded the offline suite from 140 to 158 tests.

## 1.0.0 — GameGuideLM course-project release

- Reframed the repository around grounded multi-game language modeling rather
  than a Terraria-only database application.
- Added a game-agnostic evidence contract and plug-in orchestrator.
- Integrated Terraria as the full reference knowledge module.
- Added a Stardew Valley plug-in with player-state-aware crops, fish,
  villagers, recipes, bundles, and guide retrieval.
- Added one shared Qwen3-0.6B / Qwen3-4B runtime for target, draft, and
  speculative modes.
- Added evidence-aware SFT/QLoRA data construction and optional target-teacher
  draft adaptation.
- Added token-level draft/target alignment analysis (agreement, entropy,
  top-k overlap, and Jensen–Shannon divergence).
- Added multi-game deterministic and Qwen-backed evaluation utilities.
- Added conservative citation validation and deterministic fallback.
- Preserved earlier TinyGPT, GPT-2, scheduling, and benchmark work as supporting
  course experiments rather than the final research claim.
- Added an offline release-validation command and finalized project metadata,
  licensing, documentation, and generated-artifact rules.
