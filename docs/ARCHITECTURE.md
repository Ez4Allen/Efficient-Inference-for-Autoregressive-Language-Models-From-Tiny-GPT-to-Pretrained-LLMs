# GameGuideLM Architecture

## 1. Model-facing abstraction

Every game plug-in returns a `GameGuideResult`:

```text
game, status, question, intent, entity
facts, warnings, candidates
GameEvidence[S1..Sn]
deterministic answer
```

The Qwen layer consumes only this contract. It does not contain Terraria or
Stardew-specific SQL, aliases, or progression rules.

Relevant code:

```text
src/gameguide/schemas.py
src/gameguide/plugin.py
src/gameguide/assistant.py
src/gameguide/prompting.py
src/gameguide/generator.py
src/gameguide/validation.py
```

## 2. Game plug-ins

### Terraria

```text
TerrariaGamePlugin
  → existing bilingual TerrariaAssistant
  → TerrariaFactService for Items/NPCs/Recipes/Drops
  → GuideDocumentStore for progression/mechanics
```

Terraria remains the high-coverage reference workload.

### Stardew Valley

```text
StardewAssistant
  → StardewIntentRouter
  → StardewFactService for crops/fish/villagers/recipes/bundles
  → StardewGuideStore for guide questions
  → StardewRenderer deterministic decision
```

The Stardew plug-in adds season, day, weather, time, location, bundle mode, and
other player-state conditions. This proves that the model interface is not
Terraria-specific.

## 3. Grounded generation

```text
GameGuideResult
       ↓
PromptBudgetConfig / evidence selection
       ↓
build_gameguide_prompt()
       ↓
QwenPairRuntime
       ├── Qwen3-4B target
       ├── Qwen3-0.6B draft
       └── speculative engine
       ↓
validate_gameguide_answer()
       ├── valid evidence IDs
       ├── supported URLs
       ├── supported numeric claims
       ├── no thinking trace
       └── length/empty checks
       ↓
Valid answer
  or one constrained repair pass
  or deterministic fallback
```

An explicit `UngroundedQwenGenerator` exists only for ablation. It uses the same
checkpoint without retrieved evidence and is never the safe deployed path.

The prompt builder keeps citation IDs stable while limiting source count,
per-source text, total evidence text, and large fact objects.  The validator is
restricted to the source IDs and support payload actually shown to the model;
omitted evidence cannot be cited accidentally.

## 4. Training

The existing `src/training/train_sft.py` is the only LoRA/QLoRA trainer.
GameGuideLM adds evidence-conditioned data construction, not a second trainer.

```text
Reviewed QA
  → run real retrieval
  → exact grounded prompt
  → reviewed or target-teacher answer
  → train_sft.py
```

Experiments:

- 4B evidence-following LoRA;
- 0.6B sequence-level teacher LoRA;
- future logits distillation as a separate, explicitly named implementation.

## 5. Model-pair research

The same grounded prompts support:

```text
Qwen3-4B greedy baseline
Qwen3-0.6B standalone baseline
Qwen3-0.6B → Qwen3-4B speculative decoding
```

`src/evaluation/model_pair_alignment.py` measures distributions directly:

- top-1 agreement;
- top-k overlap;
- entropy;
- Jensen-Shannon divergence;
- target-token likelihood under draft and target.

The current speculative decoder is correctness-first. Cache reuse, adaptive
draft length, and repeated warm benchmarks are future optimization stages and
must not be conflated with the v1.0.0 correctness claim.

## 6. Reliability boundaries

- Mutable facts remain outside model parameters;
- deterministic services handle calculations and conditions;
- the LLM organizes evidence rather than inventing missing facts;
- invalid generations fall back safely;
- false premises and missing player state are first-class evaluation cases;
- generated Wiki corpora and model weights are rebuilt/downloaded locally.
