# Changelog

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
