# Efficient Inference for Autoregressive Language Models

This repository contains two related engineering tracks:

1. **Autoregressive inference experiments** — tiny GPT training, prefill/decode measurement, scheduling simulation, and speculative decoding.
2. **Terraria grounded knowledge** — a reproducible structured catalog for Items, NPCs, Recipes, and Drops plus a locally indexed Official Wiki guide corpus for progression and mechanics questions.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q
```

QLoRA training requires the additional GPU dependencies:

```bash
pip install -r requirements-training.txt
```

## Terraria knowledge database

The tracked `cleaned/*.jsonl` snapshot is sufficient to rebuild all derived data:

```bash
python scripts/build_terraria_knowledge.py --quiet
```

This creates:

- `data/terraria/catalog/linked/Recipes.jsonl`
- `data/terraria/catalog/linked/Drops.jsonl`
- `data/terraria/catalog/linked/catalog_integrity_report.json`
- `data/terraria/catalog/terraria_query.sqlite3`
- `data/terraria/catalog/terraria_build_report.json`

The default build validates the tracked snapshot against
`data/terraria/catalog/snapshot_manifest.json`. Use
`--no-strict-snapshot` only when intentionally rebuilding from a newer catalog.

### Query examples

```python
from src.knowledge import TerrariaFactService

with TerrariaFactService() as service:
    recipe = service.recipe("Night's Edge")
    drops = service.drops_for_item("Beam Sword", mode="expert")
    npc = service.npc("Armored Skeleton", npc_id=77)
```


A command-line interface is also available:

```bash
python scripts/query_terraria.py recipe "Night's Edge"
python scripts/query_terraria.py drops_for_item "Beam Sword" --mode expert
python scripts/query_terraria.py npc "Armored Skeleton" --npc-id 77
```

`TerrariaQueryStore` exposes lower-level indexed queries; `TerrariaFactService`
returns compact fact packages with warnings and provenance suitable for an LLM
or API layer.

## Grounded Terraria assistant

Natural-language routing and deterministic grounded rendering are available on
top of the existing FactService:

```python
from src.assistant import TerrariaAssistant

with TerrariaAssistant(auto_build=True) as assistant:
    response = assistant.answer("How do I craft Night's Edge?")
    print(response.answer)
    print(response.evidence)
```

The first release supports Item, NPC, Recipe, reverse Recipe, Drop-source, and
source-loot questions in English and Chinese. It returns clarification for
same-name entities and refuses to invent facts for unknown entities.

```bash
python scripts/chat_terraria.py "Where can I get Beam Sword?"
python scripts/chat_terraria.py "装甲骷髅掉什么？" --mode expert
python scripts/chat_terraria.py "What is Terra Blade?" --json
```

`response.context.text` is an evidence-only prompt for an external LLM. A
custom grounded generator can be injected while the deterministic renderer
remains the fallback. See `docs/terraria_assistant.md`.

### Progression and mechanics guide corpus

The Assistant can also route progression, strategy, class-setup, arena,
housing, biome-spread, and other mechanics questions to a local document
retriever. The corpus is discovered from the Official Terraria Wiki's Guide
category plus explicitly configured core mechanics pages.

```bash
python scripts/build_terraria_guides.py
python scripts/query_terraria_guides.py   "What should I do after entering Hardmode?"
python scripts/chat_terraria.py   "进入困难模式后该做什么？"
```

The pipeline stores raw API responses, cleaned section-aware documents,
retrieval chunks, quality reports, and a local SQLite FTS5 database under
`data/terraria/guides/`. Generated text and SQLite artifacts are ignored by
Git; the source manifest and attribution file are tracked.

To prepare a lightweight review bundle after a live crawl:

```bash
python scripts/package_terraria_guide_diagnostics.py
```

Upload the resulting `terraria_guide_diagnostics.zip` for cleaning review. It
contains reports and bounded samples, not the full raw corpus.

See `docs/terraria_guides.md` for source, license, quality, and update details.

## Inference experiments

The core decoding implementations are in:

- `src/inference/autoregressive.py`
- `src/inference/speculative.py`
- `src/evaluation/`
- `src/optimization/`

A local-checkpoint speculative decoding smoke run is available as:

```bash
python scripts/smoke_speculative.py
```

It expects local GPT-2 and GPT-2 Medium checkpoints under `checkpoints/`.
Unit tests for the decoding algorithms use deterministic toy models and do not
require Hugging Face downloads.


## Benchmark workflow

Run one exact token-shape case:

```bash
python scripts/run_single_benchmark.py \
  --model gpt2 \
  --prompt-length 128 \
  --output-length 32 \
  --prompt-type technical \
  --runs 5
```

Run a resumable YAML sweep and plot the results:

```bash
python scripts/run_benchmark.py --config configs/gpt2.yaml --resume
python scripts/plot_results.py \
  --input results/raw/gpt2_benchmark.jsonl \
  --output-dir results/figures/gpt2
```

The benchmark records exact model token counts, TTFT, mean TPOT, total latency,
throughput, forward calls, and peak CUDA memory.

## TinyGPT example

```bash
python scripts/train_tiny_lm.py --config configs/tiny_gpt.yaml
python scripts/generate_tiny_lm.py \
  --checkpoint results/tiny_gpt_shakespeare/model.pt \
  --tokenizer results/tiny_gpt_shakespeare/tokenizer.json \
  --prompt "First Citizen:" \
  --max-new-tokens 200 \
  --top-k 20
```

## Serving simulation

```bash
python scripts/run_simulation.py --config configs/simulation.yaml
python scripts/plot_results.py \
  --input results/simulation/fcfs.json \
  --output-dir results/figures/simulation
```

## Terraria QLoRA smoke training

Set local model and output paths, then run the training entry point:

```bash
export TERRARIA_MODEL_PATH=/path/to/model
export TERRARIA_OUTPUT_DIR=/path/to/output
python -m src.training.train_sft --config configs/terraria_qlora_smoke.yaml
```

Relative dataset paths are resolved from the repository root. Set
`LLM_PROJECT_ROOT` to override automatic root discovery when embedding the
project elsewhere.

## Repository layout

```text
configs/                     Experiment configurations
scripts/                     Command-line entry points
src/assistant/               Grounded Terraria routing, retrieval, context, and answers
src/data/                    Dataset and prompt utilities
src/evaluation/              Benchmark and evaluation code
src/inference/               Autoregressive and speculative decoding
src/knowledge/               Terraria cleaning, linking, query, and fact APIs
src/retrieval/               Wiki import, cleaning, chunking, FTS, and guide retrieval
src/models/                  Model loaders and tiny GPT implementation
src/optimization/            Scheduling and simulation experiments
src/training/                SFT/QLoRA training
src/utils/                   Shared utilities and path discovery
tests/                       Unit and integration tests
data/terraria/catalog/       Cleaned structured snapshot and build reports
data/terraria/guides/        Guide source manifest, attribution, and generated corpus
```

## Data policy

The cleaned Terraria snapshot is tracked as a reproducible backup. Raw,
normalized, linked, and SQLite build artifacts are ignored because they can be
regenerated. The generated Wiki guide corpus is also ignored by default; its
manifest and attribution are tracked. Source attribution is recorded in
`data/terraria/catalog/ATTRIBUTION.md` and
`data/terraria/guides/ATTRIBUTION.md`.
