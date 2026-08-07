# Stardew Valley Module — Course Release v1

The Stardew module is a complete, reproducible course-project workload built on the shared GameGuideLM evidence contract. It supports deterministic bilingual question answering today and supplies the retrieval, training, evaluation, and demonstration assets required for the Qwen and speculative-decoding experiments.

## What is included

| Component | Release scope |
|---|---:|
| Structured records | 505 |
| Crops | 41 |
| Fish | 55 |
| Villagers | 34 |
| Recipes | 117 |
| Standard Bundles | 30 |
| Acquisition entities | 228 |
| Acquisition relations | 317 |
| Offline guide pages | 25 |
| Searchable guide chunks | 100 |
| Formal regression cases | 100 |
| English / Chinese cases | 50 / 50 |
| Grounded training / validation | 159 / 17 |
| Legacy SFT candidates audited | 1,262 |

The snapshot targets Stardew Valley 1.6.15 and records source URL, page title, section, version, platform, retrieval time, license, parse status, and warnings on every structured record. It is a defined course-release scope, not an assertion that the entire Wiki has been imported.

## Supported behaviors

### Structured facts

- crop season, growth time, regrowth, seed source, price, trellis, and giant-crop metadata;
- calendar-aware latest planting day, first harvest day, and harvest-count calculation;
- fish conditions by season, weather, time, location, and special restrictions;
- villager birthday and loved-gift lookup;
- cooking and crafting ingredients plus unlock source;
- complete Standard Community Center Bundle coverage;
- reverse bundle lookup by required item;
- structured acquisition sources, locations, prices, currencies, and conditions.

### Guide retrieval

The offline demonstration build contains 25 compact project-authored guide summaries. The guide pipeline cleans and chunks them into 100 FTS-indexed passages. Bilingual query expansion prioritizes topics such as first spring, Community Center, fishing, Skull Cavern, cooking progression, friendship, Greenhouse, Ginger Island, crops, and skills.

The summaries are explicitly marked `project_authored_summary`. They retain the corresponding Official Stardew Valley Wiki page URL for attribution, but they are not represented as verbatim Wiki snapshots.

### Safety and state handling

The router returns explicit statuses instead of guessing:

- `found`: sufficient structured or guide evidence exists;
- `needs_context`: required state such as season/day or current fishing conditions is missing;
- `partial`: the snapshot supports only part of the request, such as Standard but not complete Remixed Bundle data;
- `not_found`: the entity or claim is unsupported;
- `ambiguous`: more than one entity matches.

Examples:

```bash
python scripts/chat_stardew.py "How long does Parsnip take to grow?"
python scripts/chat_stardew.py "秋季第15天种Pumpkin还能收获吗？"
python scripts/chat_stardew.py "Legend在哪里、什么时候能钓？"
python scripts/chat_stardew.py "Where can I buy Return Scepter?"
python scripts/chat_stardew.py "沙漠矿洞应该怎么准备？"
python scripts/chat_stardew.py "What should I plant today?"
```

## Data layers and review boundary

### Structured release catalog

```text
data/stardew/catalog/cleaned/facts.jsonl
```

This is the runtime source for the deterministic fact database. Per-type JSONL files are exported for inspection, and `snapshot_manifest.json` records counts and SHA256 checksums.

### Grounded training set

```text
data/stardew/training/stardew_grounded_train_v1.jsonl
data/stardew/training/stardew_grounded_validation_v1.jsonl
```

These records are deterministically rendered from the structured catalog. They condition on evidence and require citations. The formal evaluation files are excluded by construction.

### Formal regression suite

```text
data/stardew/evaluation/stardew_validation_v1.jsonl   # 40
data/stardew/evaluation/stardew_eval_v1.jsonl         # 60
```

Distribution:

- language: 50 English / 50 Chinese;
- category: 20 crop, 15 fish, 15 villager, 15 recipe, 15 bundle, 10 acquisition, 10 guide;
- status: 70 found, 10 needs_context, 10 partial, 10 not_found.

The deterministic system passes all 100 cases. These are regression candidates generated and checked by code, not an independently human-approved research benchmark. Every row therefore remains:

```json
{
  "review_status": "machine_validated",
  "reviewer": null,
  "human_review_required": true
}
```

### Legacy teammate SFT pool

The original 1,262 AI-assisted English records are retained for traceability but no longer claim universal verification. The cleanup pipeline canonicalizes sources, maps 490 free-form categories to controlled intents, creates source/template-disjoint development splits, and sets every row to:

```json
{
  "verified": false,
  "review_status": "pending",
  "reviewed_by": null
}
```

They are not used as formal evaluation data and should not be used for model training until reviewed.

## One-command release build

```bash
python scripts/build_stardew_release.py
```

The command performs:

1. deterministic data generation;
2. legacy SFT cleanup and leakage audit;
3. structured SQLite build;
4. offline guide SQLite build;
5. 100-case deterministic evaluation;
6. release-contract validation;
7. showcase output generation;
8. static HTML dashboard generation;
9. the complete repository test suite.

Expected offline result:

```text
100 / 100 deterministic Stardew regression cases passed
184 repository tests passed
SQLite integrity: ok
Release readiness: engineering_passed_human_review_pending
```

## Presentation assets

- `demo/stardew_showcase.html` — self-contained interactive dashboard;
- `results/stardew/demo_outputs.md` — curated bilingual demonstration transcript;
- `results/stardew/evaluation_summary.json` — slice metrics;
- `results/stardew/release_validation.json` — release-contract evidence;
- `results/stardew/release_build_manifest.json` — build steps, timings, hashes, and truth boundary.

## Model integration

The deterministic module supplies the exact evidence package used by:

- Qwen3-4B target-only grounded generation;
- optional QLoRA adaptation for citation and evidence following;
- Qwen3-0.6B speculative draft baseline;
- custom `TinyQwenDraft` target-teacher training and speculative decoding.

No trained Qwen adapter or TinyQwenDraft checkpoint is claimed in the offline release. Model-quality and GPU-speed claims require a real training and warm benchmark run.
