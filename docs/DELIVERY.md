# GameGuideLM v1.0.0 Delivery

## What this release is

GameGuideLM is a model-centric multi-game question-answering research system.
It studies how Qwen can answer game-guide questions from versioned external
evidence, how evidence-aware LoRA can improve grounding behavior, and how a
Qwen3-0.6B draft aligns with a Qwen3-4B target for speculative decoding.

Terraria and Stardew Valley are the two evaluation workloads. Their schemas and
retrieval rules are game-specific, while prompting, Qwen generation, grounding
validation, LoRA data construction, model-pair analysis, and decoding are
shared.

## End-to-end pipeline

```text
Question + game + optional player state
                  ↓
          Game-specific plug-in
        ┌─────────┴──────────┐
        ↓                    ↓
Structured FactService   Guide retriever
        └─────────┬──────────┘
                  ↓
       Common evidence contract
                  ↓
  Prompt-budgeted evidence selection
                  ↓
      Evidence-conditioned Qwen prompt
                  ↓
 target / draft / speculative generation
                  ↓
 citation, URL, numeric and trace validation
                  ↓
 valid answer / repair / deterministic fallback
```

## Model experiments

- Qwen3-4B autoregressive target baseline;
- Qwen3-0.6B standalone draft baseline;
- correctness-first Qwen3-0.6B → Qwen3-4B speculative decoding;
- evidence-aware 4B LoRA data and configuration;
- sequence-level 0.6B teacher adaptation data and configuration;
- cache-aware completion-logit analysis: top-1 agreement, top-k overlap,
  entropy, target-token likelihood, and Jensen-Shannon divergence;
- warm repeated target/draft/speculative benchmark with TTFT, TPOT, latency,
  throughput, forward calls, acceptance, output hashes, and exact-match checks.

## Included game scope

### Terraria

The tracked structured snapshot contains 6,283 Items, 770 NPCs, 3,409 Recipes,
4,221 Recipe variants, and 3,144 Drops. The build validates 14,353 resolved
references. A configurable Official Wiki guide pipeline supplies progression
and mechanics evidence.

### Stardew Valley

The release includes a compact, source-linked 31-record integration snapshot
covering representative crops, fish, villagers, crafting recipes, and Standard
Bundles. It also includes four curated offline guide seed documents and a full
online Wiki guide pipeline. The compact snapshot is an integration seed, not a
claim of complete Stardew Wiki coverage; the teammate-maintained module can
extend the same contracts without changing model code.

## Verified release state

```text
Python test suite:                         140 passed
Terraria structured rebuild:              passed
Terraria resolved references:             14,353
Stardew structured rebuild:               passed
Stardew tracked starter facts:             31
Stardew offline guide seed:                 4 documents / 16 chunks
SQLite integrity and smoke queries:        passed
```

Reproduce the offline release validation with:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
python scripts/validate_release.py
```

## Claim boundaries

- No Qwen weights or trained LoRA adapters are distributed.
- The LoRA configurations and evidence-conditioned data builders are included,
  but model-quality gains require actual GPU training and evaluation.
- The speculative decoder is a correctness-first baseline; the release does
  not claim a speedup without warm repeated GPU measurements.
- Online Wiki corpora are not distributed and must be built locally.
- Automated lexical metrics are regression signals, not substitutes for human
  factuality and usefulness review.
