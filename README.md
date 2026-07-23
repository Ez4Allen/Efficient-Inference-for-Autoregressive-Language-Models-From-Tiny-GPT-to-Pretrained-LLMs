# Efficient Inference for Autoregressive Language Models

This repository contains two related engineering tracks:

1. **Autoregressive inference experiments** — tiny GPT training, prefill/decode measurement, scheduling simulation, and speculative decoding.
2. **Terraria structured knowledge** — a reproducible catalog pipeline that links Items, NPCs, Recipes, and Drops, validates references, builds an indexed SQLite database, and exposes deterministic query APIs.

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
src/data/                    Dataset and prompt utilities
src/evaluation/              Benchmark and evaluation code
src/inference/               Autoregressive and speculative decoding
src/knowledge/               Terraria cleaning, linking, query, and fact APIs
src/models/                  Model loaders and tiny GPT implementation
src/optimization/            Scheduling and simulation experiments
src/training/                SFT/QLoRA training
src/utils/                   Shared utilities and path discovery
tests/                       Unit and integration tests
data/terraria/catalog/       Cleaned snapshot, manifest, and build reports
```

## Data policy

The cleaned Terraria snapshot is tracked as a reproducible backup. Raw,
normalized, linked, and SQLite build artifacts are ignored because they can be
regenerated. Source attribution is recorded in
`data/terraria/catalog/ATTRIBUTION.md`.
